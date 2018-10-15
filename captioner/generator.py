import math
import torch
import torch.nn.functional as F


# Copyright (c) 2017-present, Facebook, Inc.
class SequenceGenerator(object):
    def __init__(
        self, model, dictionary, beam_size=1, minlen=1, maxlen=None, stop_early=True, normalize_scores=True,
        len_penalty=1, unk_penalty=0,
    ):
        """Generates translations of a given source sentence.
        Args:
            min/maxlen: The length of the generated output will be bounded by minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after finalizing beam_size hypotheses, though longer hypotheses might have better normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.model = model
        self.pad = dictionary.pad_idx
        self.unk = dictionary.unk_idx
        self.eos = dictionary.eos_idx
        self.vocab_size = len(dictionary)
        self.beam_size = beam_size
        self.minlen = minlen
        self.maxlen = maxlen
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty

    def generate(self, image_features):
        """Generate a batch of translations."""
        with torch.no_grad():
            return self._generate(image_features)

    def _generate(self, image_features):
        bsz = image_features.size(0)
        maxlen, beam_size = self.maxlen, self.beam_size
        self.model.eval()

        image_features = image_features.repeat(1, beam_size, 1).view(-1, *image_features.size()[1:])
        incremental_states = {}

        # Initialize buffers
        scores = image_features.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = image_features.data.new(bsz * beam_size, maxlen + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos
        attn, attn_buf = None, None

        # List of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # Number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # Offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # Helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by comparing the worst score
            among finalized hypotheses to the best possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # Stop if the best unfinalized score is worse than the worst finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total number of finalized hypotheses
            per sentence <= beam_size. Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those that appear later.
            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size), indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing scores for each hypothesis
                unfinalized_scores: A vector containing scores for all unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # Clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1: step + 2] if attn is not None else None

            # Compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # Convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # Normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                sent = idx // beam_size
                sents_seen.add(sent)

                def get_hypo():
                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': attn_clone[i] if attn_clone is not None else None,  # src_len x tgt_len
                        'alignment': attn_clone[i].max(dim=0)[1] if attn_clone is not None else None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # Replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # Find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {'score': s['score'], 'idx': idx}

            num_finished = 0
            for sent in sents_seen:
                # Check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    num_finished += 1
            return num_finished

        reorder_state = None
        batch_idxs = None
        for step in range(maxlen + 1):  # Extra step for EOS marker
            # Reorder decoder internal states based on the previous choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # Update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)

                self.model.reorder_incremental_state(incremental_states, reorder_state)

            lprobs, avg_attn_scores = self._decode(tokens[:, :step + 1], image_features, incremental_states, log_probs=True)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, image_features.size(1), maxlen + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)

            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                torch.topk(
                    lprobs.view(bsz, -1),
                    k=min(cand_size, lprobs.view(bsz, -1).size(1) - 1),  # -1 so we never select pad
                    out=(cand_scores, cand_indices),
                )
                torch.div(cand_indices, self.vocab_size, out=cand_beams)
                cand_indices.fmod_(self.vocab_size)
            else:
                # Make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # Finalize all active hypotheses once we hit maxlen
                # Pick the hypothesis with the highest prob of EOS right now
                torch.sort(lprobs[:, self.eos], descending=True, out=(eos_scores, eos_bbsz_idx))
                num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate hypotheses, with a range of
            # values: [0, bsz*beam_size), and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # Finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size], out=eos_bbsz_idx, )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size], out=eos_scores)
                    num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[:eos_mask.size(1)], out=active_mask)

            # Get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(active_mask, k=beam_size, dim=1, largest=False, out=(_ignore, active_hypos))
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(cand_bbsz_idx, dim=1, index=active_hypos, out=active_bbsz_idx)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos, out=scores[:, step].view(bsz, beam_size))
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # Copy tokens and scores for active hypotheses
            torch.index_select(tokens[:, :step + 1], dim=0, index=active_bbsz_idx, out=tokens_buf[:, :step + 1])
            torch.gather(cand_indices, dim=1, index=active_hypos, out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1])
            if step > 0:
                torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx, out=scores_buf[:, :step])
            torch.gather(cand_scores, dim=1, index=active_hypos, out=scores_buf.view(bsz, beam_size, -1)[:, :, step])

            # Copy attention for active hypotheses
            if attn is not None:
                torch.index_select(attn[:, :, :step + 2], dim=0, index=active_bbsz_idx, out=attn_buf[:, :, :step + 2])

            # Swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # Reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # Sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        return finalized

    def _decode(self, tokens, image_features, incremental_states, log_probs):
        with torch.no_grad():
            decoder_out = list(self.model(image_features, tokens, incremental_state=incremental_states))
            decoder_out[0] = decoder_out[0][:, -1, :]
            attn = decoder_out[1]
            if attn is not None:
                attn = attn[:, -1, :]

        if log_probs:
            probs = F.log_softmax(decoder_out[0], dim=1)
        else:
            probs = F.softmax(decoder_out[0], dim=1)
        return probs, attn
