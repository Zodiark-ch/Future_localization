import os
import sys
from copy import deepcopy
from time import time

sys.path.append("src")
import datasets
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer import is_datasets_available
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from pruner.utils import WrappedGPT, find_layers
from training_losses import task_loss


class GenerateMask(Trainer):
    def __init__(self, score_type, ratios, mask_dir, p, q,mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_type = score_type
        self.ratios = ratios if ratios else [0.0, 0.1, 0.2, 0.5, 0.8, 0.9]
        self.mask_dir = mask_dir
        self.p = p
        self.q = q
        self.mu = mu

    def _get_batch_data(self, inputs, key):
        if key in inputs and inputs[key] is not None:
            return inputs[key]
        for candidate, value in inputs.items():
            if candidate.startswith(key) and value is not None:
                return value
        if key == "pervasiveness":
            for candidate, value in inputs.items():
                if candidate.startswith("pervasiveness") and value is not None:
                    return value
        raise KeyError(f"No batch data found for {key}")

    def score2mask(self, scores, ratio, return_rank=False):
        if return_rank:
            positions = torch.argsort(scores)
            ranks = torch.argsort(positions)
            return ranks
        hard_dict = self._build_flat_score_mask(scores, ratio)
        print(f"Generated mask with {len(hard_dict)} parameters")
        return hard_dict

    def get_mask(self):
        if self.score_type == "gradient":
            return self.gradient()
        elif self.score_type == "gradient_vis":
            self.gradient(save_masks=False, keep_scores=True)
            torch.save(self.scores, os.path.join(self.mask_dir, f"scores.pt"))
            exit(0)
        elif self.score_type == "weight":
            self.weight()
        elif self.score_type == "weight_vis":
            self.weight()
            torch.save(self.scores, os.path.join(self.mask_dir, f"scores.pt"))
            exit(0)
        elif self.score_type == "random":
            self.random()
        elif self.score_type == "snip_advanced":
            self.snip_advanced()
        elif self.score_type == "snip_advanced_CL":
            self.snip_advanced(CL=True)
        elif self.score_type == "snip_advanced_gn":
            self.snip_advanced_gn()
        elif self.score_type == "snip_advanced_visualization":
            self.snip_advanced_visualization()
        elif self.score_type == "snip_advanced_new":
            self.snip_advanced_new()
        elif self.score_type == "FFN":
            hard_dict = {}
            for named, tensor in self.model.named_parameters():
                if "fc" in named or "final_layer_norm" in named:
                    hard_dict[named] = torch.ones_like(tensor)
                else:
                    hard_dict[named] = torch.zeros_like(tensor)
            torch.save(hard_dict, os.path.join(self.mask_dir, f"with_0.0.pt"))
            return
        elif self.score_type == "wanda":
            self.wanda()
            return
        else:
            raise ValueError(f"score_type {self.score_type} not supported")
        
        # Save the mask
        
        for ratio in self.ratios:
            if os.path.exists(os.path.join(self.mask_dir, f"with_{ratio}.pt")):
                continue
            hard_dict = self._build_flat_score_mask(self.scores, ratio)
            torch.save(hard_dict, os.path.join(self.mask_dir, f"with_{ratio}.pt"))

    def _parameter_items(self):
        return list(self.model.named_parameters())

    def _build_flat_score_mask(self, scores, ratio):
        scores = scores.detach().cpu().flatten()
        total_count = scores.numel()
        keep_count = int(total_count * ratio)

        if keep_count <= 0:
            return {
                key: torch.zeros(tensor.shape, dtype=torch.bool)
                for key, tensor in self._parameter_items()
            }
        if keep_count >= total_count:
            return {
                key: torch.ones(tensor.shape, dtype=torch.bool)
                for key, tensor in self._parameter_items()
            }

        threshold = torch.kthvalue(scores, keep_count).values.item()
        hard_dict = {}
        start_index = 0
        kept_count = 0
        for key, tensor in self._parameter_items():
            num_elements = tensor.numel()
            tensor_scores = scores[start_index : start_index + num_elements]
            threshold_tensor = tensor_scores < threshold
            kept_count += int(threshold_tensor.sum().item())
            hard_dict[key] = threshold_tensor.reshape(tensor.shape).type(torch.bool)
            start_index += num_elements

        remaining = keep_count - kept_count
        if remaining > 0:
            start_index = 0
            for key, tensor in self._parameter_items():
                if remaining <= 0:
                    break
                num_elements = tensor.numel()
                tensor_scores = scores[start_index : start_index + num_elements]
                candidates = tensor_scores == threshold
                filled = self._fill_mask_from_candidates(
                    hard_dict[key], candidates.reshape(tensor.shape), remaining
                )
                remaining -= filled
                kept_count += filled
                start_index += num_elements

        print(
            f"Generated mask ratio={ratio}: kept={kept_count}/{total_count} "
            f"({kept_count / max(1, total_count):.6f}), threshold={threshold:.6e}"
        )
        return hard_dict

    def _score_total_count(self, score_tensors):
        return sum(tensor.numel() for tensor in score_tensors.values())

    def _score_min_max(self, score_tensors):
        min_score = None
        max_score = None
        for tensor in score_tensors.values():
            if tensor.numel() == 0:
                continue
            tensor_min = float(tensor.min().item())
            tensor_max = float(tensor.max().item())
            min_score = tensor_min if min_score is None else min(min_score, tensor_min)
            max_score = tensor_max if max_score is None else max(max_score, tensor_max)
        return min_score, max_score

    def _count_scores_above(self, score_tensors, threshold):
        total = 0
        for tensor in score_tensors.values():
            total += int((tensor > threshold).sum().item())
        return total

    def _find_keep_threshold(self, score_tensors, keep_count):
        min_score, max_score = self._score_min_max(score_tensors)
        if min_score is None or max_score is None or min_score == max_score:
            return min_score if min_score is not None else 0.0

        low = min_score
        high = max_score
        for _ in range(32):
            mid = (low + high) / 2.0
            if self._count_scores_above(score_tensors, mid) >= keep_count:
                low = mid
            else:
                high = mid
        return low

    def _fill_mask_from_candidates(self, mask, candidates, remaining, chunk_size=1_000_000):
        if remaining <= 0:
            return 0
        flat_mask = mask.view(-1)
        flat_candidates = candidates.view(-1)
        filled = 0
        for start in range(0, flat_candidates.numel(), chunk_size):
            if filled >= remaining:
                break
            end = min(start + chunk_size, flat_candidates.numel())
            candidate_chunk = flat_candidates[start:end]
            candidate_count = int(candidate_chunk.sum().item())
            if candidate_count == 0:
                continue
            take = min(remaining - filled, candidate_count)
            if take == candidate_count:
                flat_mask[start:end] |= candidate_chunk
            else:
                candidate_indices = candidate_chunk.nonzero(as_tuple=False).flatten()[:take]
                flat_mask[start + candidate_indices] = True
            filled += take
        return filled

    def _build_threshold_mask(self, score_tensors, ratio):
        total_count = self._score_total_count(score_tensors)
        keep_count = int(total_count * ratio)
        if keep_count <= 0:
            return {
                key: torch.zeros(tensor.shape, dtype=torch.bool)
                for key, tensor in score_tensors.items()
            }
        if keep_count >= total_count:
            return {
                key: torch.ones(tensor.shape, dtype=torch.bool)
                for key, tensor in score_tensors.items()
            }

        threshold = self._find_keep_threshold(score_tensors, keep_count)
        greater_count = self._count_scores_above(score_tensors, threshold)
        hard_dict = {}

        if greater_count >= keep_count:
            remaining = keep_count
            for key, tensor in score_tensors.items():
                mask = torch.zeros(tensor.shape, dtype=torch.bool)
                remaining -= self._fill_mask_from_candidates(
                    mask, tensor > threshold, remaining
                )
                hard_dict[key] = mask
            kept_count = keep_count - remaining
        else:
            remaining = keep_count - greater_count
            kept_count = greater_count
            for key, tensor in score_tensors.items():
                mask = tensor > threshold
                hard_dict[key] = mask
            for key, tensor in score_tensors.items():
                if remaining <= 0:
                    break
                filled = self._fill_mask_from_candidates(
                    hard_dict[key], tensor <= threshold, remaining
                )
                remaining -= filled
                kept_count += filled

        print(
            f"Generated gradient mask ratio={ratio}: kept={kept_count}/{total_count} "
            f"({kept_count / max(1, total_count):.6f}), threshold={threshold:.6e}"
        )
        return hard_dict

    def _save_masks_from_score_tensors(self, score_tensors):
        for ratio in self.ratios:
            mask_path = os.path.join(self.mask_dir, f"with_{ratio}.pt")
            if os.path.exists(mask_path):
                continue
            hard_dict = self._build_threshold_mask(score_tensors, ratio)
            torch.save(hard_dict, mask_path)
            del hard_dict

    def gradient(self, dataset="target", save_masks=True, keep_scores=False):
        gradient_scores = {}

        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        mask_parameter_items = self._parameter_items()
        mask_parameter_names = [name for name, _ in mask_parameter_items]
        model.zero_grad(set_to_none=True)
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {dataset} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, dataset)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, (_, tensor) in zip(mask_parameter_names, model.named_parameters()):
                    grad = tensor.grad
                    if grad is None:
                        continue
                    grad_cpu = grad.detach().to(device="cpu", dtype=torch.float32)
                    if key not in gradient_scores:
                        gradient_scores[key] = grad_cpu
                    else:
                        gradient_scores[key].add_(grad_cpu)

            model.zero_grad(set_to_none=True)

        for key, tensor in mask_parameter_items:
            if key not in gradient_scores:
                gradient_scores[key] = torch.zeros(tensor.shape, dtype=torch.float32)
            else:
                gradient_scores[key].abs_()

        if keep_scores:
            self.scores = torch.cat(
                [-scores.flatten() for scores in gradient_scores.values()]
            )
            return None
        if save_masks:
            self._save_masks_from_score_tensors(gradient_scores)
        del gradient_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    def weight(self):
        begin_time = time()
        weights = {}
        with torch.no_grad():
            for key, tensor in self.model.named_parameters():
                weights[key] = -torch.abs(tensor.data)
            end_time = time()  
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat([weight.flatten().cpu() for weight in weights.values()])

    def random(self):
        begin_time = time()
        random = {}
        with torch.no_grad():
            for key, tensor in self.model.named_parameters():
                random[key] = torch.rand_like(tensor.data)
            end_time = time()  
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat(
                [random.flatten().cpu() for random in random.values()]
            )

    def hessianfree(self, pervasiveness_epoch=0, CL=False):
        hessianfree = {}
        pervasiveness_grad = {}
        for key, tensor in self.model.named_parameters():
            hessianfree[key] = 0
            pervasiveness_grad[key] = 0
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in hessianfree:
                        hessianfree[key] = tensor.grad.data
                    else:
                        hessianfree[key] += tensor.grad.data

            model.zero_grad()

        model.zero_grad()
        if pervasiveness_epoch > 0:
            epochs = pervasiveness_epoch
        else:
            epochs = 1
        for _ in range(epochs):
            for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in pervasiveness_grad:
                            pervasiveness_grad[key] = tensor.grad.data
                        else:
                            pervasiveness_grad[key] += tensor.grad.data
                model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                hessianfree[key] = -hessianfree[key] * pervasiveness_grad[key]

            self.scores = torch.cat(
                [hessianfree.flatten().cpu() for hessianfree in hessianfree.values()]
            )

    def hessianfree_smooth(self, T=10, sigma=1e-5):
        hessianfree = {}
        pervasiveness_grad = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for t in range(T):
            noise_t = {}
            hessianfree = {}
            pervasiveness_grad = {}
            for name, tensor in model.named_parameters():
                noise_t[name] = torch.randn_like(tensor) * sigma
                tensor.data += noise_t[name]
            for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "target")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in hessianfree:
                            hessianfree[key] = tensor.grad.data
                        else:
                            hessianfree[key] += tensor.grad.data

                model.zero_grad()
            for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in pervasiveness_grad:
                            pervasiveness_grad[key] = tensor.grad.data
                        else:
                            pervasiveness_grad[key] += tensor.grad.data
                model.zero_grad()
            with torch.no_grad():
                if t == 0:
                    for key, tensor in model.named_parameters():
                        hessianfree[key] = -torch.abs(hessianfree[key] * pervasiveness_grad[key]) / T
                else:
                    for key, tensor in model.named_parameters():
                        hessianfree[key] += -torch.abs(hessianfree[key] * pervasiveness_grad[key]) / T
            for name, tensor in model.named_parameters():
                tensor.data -= noise_t[name]
        with torch.no_grad():
            self.scores = torch.cat(
                [hessianfree.flatten().cpu() for hessianfree in hessianfree.values()]
            )


    def gradient_smooth(self, T=10, sigma=1e-5):
        scores = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for t in range(T):
            noise_t = {}
            gradient = {}
            for name, tensor in model.named_parameters():
                noise_t[name] = torch.randn_like(tensor) * sigma
                tensor.data += noise_t[name]
            for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "target")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in gradient:
                            gradient[key] = tensor.grad.data
                        else:
                            gradient[key] += tensor.grad.data

                model.zero_grad()
            if t == 0:
                for key, tensor in model.named_parameters():
                    scores[key] = -torch.abs(gradient[key]) / T
            else:
                for key, tensor in model.named_parameters():
                    scores[key] += -torch.abs(gradient[key]) / T
            for name, tensor in model.named_parameters():
                tensor.data -= noise_t[name]
        with torch.no_grad():
            self.scores = torch.cat(
                [gradient.flatten().cpu() for gradient in gradient.values()]
            )

    def hessian(self):
        mu = 1e-3
        target_gradient = {}
        snip_hessian = {}
        pervasiveness_gradient_all = {}
        hessian = {}
        beta = 0.95
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in target_gradient:
                        target_gradient[key] = tensor.grad.data
                    else:
                        target_gradient[key] += tensor.grad.data

            model.zero_grad()
        for i in range(10):
            pervasiveness_gradient = {}
            for inputs in tqdm.tqdm(train_dataloader, desc="computing hessian"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in pervasiveness_gradient:
                            pervasiveness_gradient[key] = tensor.grad.data
                        else:
                            pervasiveness_gradient[key] += tensor.grad.data
                        if key not in pervasiveness_gradient_all:
                            pervasiveness_gradient_all[key] = tensor.grad.data
                        else:
                            pervasiveness_gradient_all[key] += tensor.grad.data
                model.zero_grad()
            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in hessian:
                        hessian[key] = pervasiveness_gradient[key] * pervasiveness_gradient[key]
                    else:
                        hessian[key] = (
                            beta * hessian[key]
                            + (1 - beta) * pervasiveness_gradient[key] * pervasiveness_gradient[key]
                        )
            model.zero_grad()

        with torch.no_grad():
            for key, tensor in model.named_parameters():
                snip_hessian[key] = -torch.abs(
                    mu
                    * (tensor.data - pervasiveness_gradient_all[key] / hessian[key])
                    * target_gradient[key]
                    - mu
                    * mu
                    * pervasiveness_gradient_all[key]
                    / hessian[key]
                    * target_gradient[key]
                )

            self.scores = torch.cat(
                [snip_hessian.flatten().cpu() for snip_hessian in snip_hessian.values()]
            )

    def snip(self, name, CL = False, FT = False, layer_wise = False):
        gradient = {}
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {name} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, name, CL=CL, FT=FT)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in gradient:
                        gradient[key] = tensor.grad.data
                    else:
                        gradient[key] += tensor.grad.data

            model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                snip[key] = -torch.abs(gradient[key] * tensor.data)
        if layer_wise:
            for ratio in self.ratios:
                W_masks = {}
                for key, tensor in model.named_parameters():
                    k = int(ratio * tensor.numel())
                    flat_tensor = tensor.flatten()
                    # Get the top k values and their indices from the flattened tensor
                    _, top_k_indices = torch.topk(flat_tensor, k, largest=True)
                    # Create a boolean mask for the flattened tensor
                    flat_mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
                    # Set True in the positions of the top k elements
                    flat_mask[top_k_indices] = True
                    # Reshape the flat mask back to the original tensor shape
                    mask = flat_mask.reshape(tensor.shape)
                    W_masks[key] = mask
                torch.save(W_masks, os.path.join(self.mask_dir, f"with_{ratio}.pt"))
            return
        else:
            with torch.no_grad():
                scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
            return scores

    def snip_target_reinit(self, name):
        gradient = {}
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {name} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, name)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in gradient:
                        gradient[key] = tensor.grad.data
                    else:
                        gradient[key] += tensor.grad.data

            model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                snip[key] = -torch.abs(gradient[key] * tensor.data)
        with torch.no_grad():
            scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
        mask = self.score2mask(scores, self.p, return_rank=False)
        del scores
        total = 0
        non_zero = 0
        with torch.no_grad():
            for key,tensor in model.named_parameters():
                total += tensor.numel()
                non_zero += mask[key].sum().item()
                tensor.data = tensor.data * (1-mask[key].to(tensor.device)) + torch.randn_like(tensor) * mask[key].to(tensor.device)
        print(f"total non zero: {non_zero}")
        print(f"Sparsity: {1-non_zero/total}")
        return






    def snip_smooth(self, name,T=10, sigma=1e-5):
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for t in range(T):
            noise_t = {}
            gradient = {}
            for name, tensor in model.named_parameters():
                noise_t[name] = torch.randn_like(tensor) * sigma
                tensor.data += noise_t[name]            
            for inputs in tqdm.tqdm(train_dataloader, desc=f"computing target gradient"):
                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_adapted(model, inputs, "target")

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.accelerator.backward(loss)

                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        if key not in gradient:
                            gradient[key] = tensor.grad.data
                        else:
                            gradient[key] += tensor.grad.data

                model.zero_grad()
            with torch.no_grad():
                if t == 0:
                    for key, tensor in model.named_parameters():
                        snip[key] = -torch.abs(gradient[key] * tensor.data)/T
                else:
                    for key, tensor in model.named_parameters():
                        snip[key] += -torch.abs(gradient[key] * tensor.data)/T
            for name, tensor in model.named_parameters():
                tensor.data -= noise_t[name]
                    
        with torch.no_grad():
            scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
        return scores



    def snip_visualization(self, name):
        snip = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc=f"computing {name} gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, name)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in snip:
                        snip[key] = -torch.abs(tensor.grad.data * tensor.data)
                    else:
                        snip[key] += -torch.abs(tensor.grad.data * tensor.data)

            model.zero_grad()
        with torch.no_grad():
            scores = torch.cat([snip.flatten().cpu() for snip in snip.values()])
        torch.save(scores, os.path.join(self.mask_dir, f"scores.pt"))
        exit(0)

    def snip_advanced(self, CL=False):
        begin_time = time()
        target_gradint = {}
        pervasiveness_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in target_gradint:
                        target_gradint[key] = tensor.grad.data
                    else:
                        target_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in pervasiveness_gradint:
                        pervasiveness_gradint[key] = tensor.grad.data
                    else:
                        pervasiveness_gradint[key] += tensor.grad.data
            model.zero_grad()

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (tensor.data - pervasiveness_gradint[key] / self.mu) * target_gradint[key]
                )
            end_time = time()  
            print(f"Mask generation Time taken: {end_time-begin_time}")
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )


    def snip_advanced_gn(self, CL=False):
        target_gradint = {}
        pervasiveness_gradint = {}
        hessian = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target", CL=CL)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in target_gradint:
                        target_gradint[key] = tensor.grad.data
                    else:
                        target_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in pervasiveness_gradint:
                        pervasiveness_gradint[key] = tensor.grad.data
                    else:
                        pervasiveness_gradint[key] += tensor.grad.data
            model.zero_grad()
        total_hessian = 0
        num_hessian = 0
        for inputs in tqdm.tqdm(train_dataloader, desc="computing hessian"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)
            cnt = 0
            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in hessian:
                        hessian[key] = tensor.grad.data 
                    else:
                        hessian[key] += tensor.grad.data
                cnt += 1
            model.zero_grad()
        for key in hessian.keys():
            hessian[key] = hessian[key] * hessian[key]/(cnt*cnt)
            num_elements = hessian[key].numel()
            num_hessian += num_elements
            total_hessian += hessian[key].sum().item()
        print(f"mean hessian: {total_hessian/num_hessian}")            

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (tensor.data - pervasiveness_gradint[key] / hessian[key]) * target_gradint[key]
                )
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )


    def snip_advanced_visualization(self):
        mu = 1e-3
        target_gradint = {}
        pervasiveness_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in target_gradint:
                        target_gradint[key] = tensor.grad.data
                    else:
                        target_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in pervasiveness_gradint:
                        pervasiveness_gradint[key] = tensor.grad.data
                    else:
                        pervasiveness_gradint[key] += tensor.grad.data
            model.zero_grad()

        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    mu * (tensor.data - pervasiveness_gradint[key]) * target_gradint[key]
                    - mu * mu * pervasiveness_gradint[key] * target_gradint[key]
                )
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )
        torch.save(self.scores, os.path.join(self.mask_dir, f"scores.pt"))
        exit(0)

    def snip_advanced_new(self):
        target_gradint = {}
        pervasiveness_gradint = {}
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()
        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in target_gradint:
                        target_gradint[key] = tensor.grad.data
                    else:
                        target_gradint[key] += tensor.grad.data

            model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    if key not in pervasiveness_gradint:
                        pervasiveness_gradint[key] = tensor.grad.data
                    else:
                        pervasiveness_gradint[key] += tensor.grad.data
            model.zero_grad()
        mu = 1e-3
        with torch.no_grad():
            scores = {}
            for key, tensor in model.named_parameters():
                scores[key] = -torch.abs(
                    (
                        mu
                        - (mu / (tensor.data + 1e-12))
                        * (1 + mu / (tensor.data + 1e-12))
                        * pervasiveness_gradint[key]
                    )
                    * target_gradint[key]
                )
            self.scores = torch.cat(
                [scores.flatten().cpu() for scores in scores.values()]
            )

    def normalizedhf(self):
        hessianfree = {}
        pervasiveness_grad = {}
        for key, tensor in self.model.named_parameters():
            hessianfree[key] = 0
            pervasiveness_grad[key] = 0
        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing target gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "target")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    hessianfree[key] += tensor.grad.data.cpu()

            model.zero_grad()

        model.zero_grad()
        for inputs in tqdm.tqdm(train_dataloader, desc="computing pervasiveness gradient"):
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_adapted(model, inputs, "pervasiveness")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            with torch.no_grad():
                for key, tensor in model.named_parameters():
                    pervasiveness_grad[key] += tensor.grad.data.cpu()
            model.zero_grad()
        with torch.no_grad():
            for key, tensor in model.named_parameters():
                hessianfree[key] = (
                    -hessianfree[key]
                    * pervasiveness_grad[key]
                    / (torch.abs(tensor.data.cpu()) + 1e-8)
                )
            self.scores = torch.cat(
                [hessianfree.flatten() for hessianfree in hessianfree.values()]
            )

    def wanda(self):
        begin_time = time()
        W_metrics = {}

        self.accelerator.free_memory()
        self.model.eval()
        train_dataloader = self.get_train_dataloader()

        model = self._wrap_model(self.model)

        model, self.optimizer = self.accelerator.prepare(model, self.optimizer)

        use_cache = model.config.use_cache
        model.config.use_cache = False

        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = self.prepare_calibration_input(
                model, train_dataloader
            )
        try:
            layers = model.model.layers
        except:
            layers = model.model.decoder.layers
        cnt = 0
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            if f"model.layers.{i}" in model.hf_device_map:
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev),
                    outs.to(dev),
                    attention_mask.to(dev),
                    position_ids.to(dev),
                )

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(len(train_dataloader)):
                with torch.no_grad():
                    if position_ids is not None:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
            for handle in handles:
                handle.remove()

            for name in subset:
                W_metrics[cnt] = (
                    (
                        torch.abs(subset[name].weight.data)
                        * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    )
                    .detach()
                    .cpu()
                )
                cnt += 1
            for j in range(len(train_dataloader)):
                with torch.no_grad():
                    if position_ids is not None:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )[0]
                    else:
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                        )[0]
            inps, outs = outs, inps
        end_time = time()
        print(f"Mask generation Time taken: {end_time-begin_time}")
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()
        for ratio in self.ratios:
            W_masks = {}
            cnt = 0
            for i in range(len(layers)):
                layer = layers[i]
                subset = find_layers(layer)

                for name in subset:
                    W_metric = W_metrics[cnt]
                    W_mask = torch.zeros_like(W_metric) == 1
                    sort_res = torch.sort(W_metric, dim=-1, stable=True) # sort the weights into ascending order
                    indices = sort_res[1][:, : int(W_metric.shape[1] * (1-ratio))]
                    W_mask.scatter_(1, indices, True)
                    W_masks[cnt] = ~W_mask
                    cnt += 1
            for i in range(len(W_masks)):
                W_masks[i] = W_masks[i].type(torch.bool)
            torch.save(W_masks, os.path.join(self.mask_dir, f"with_{ratio}.pt"))

    def prepare_calibration_input(self, model, dataloader):
        use_cache = model.config.use_cache
        model.config.use_cache = False
        try:
            layers = model.model.layers
        except:
            layers = model.model.decoder.layers
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
        else:
            device = next(iter(model.parameters())).device
        dtype = next(iter(model.parameters())).dtype
        batch = next(iter(dataloader))
        batch = self._get_batch_data(batch, "target")
        inps = torch.zeros(
            (len(dataloader), batch[0].shape[1], model.config.hidden_size),
            dtype=dtype,
            device=device,
        )
        inps.requires_grad = False
        cache = {"i": 0, "attention_mask": None, "position_ids": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                try:
                    cache["position_ids"] = kwargs["position_ids"]
                except:
                    pass
                raise ValueError

        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            batch = self._get_batch_data(batch, "target")
            try:
                model(input_ids=batch[0], attention_mask=batch[1])
            except ValueError:
                pass
        layers[0] = layers[0].module
        outs = torch.zeros_like(inps)
        attention_mask = cache["attention_mask"]
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, batch[0].shape[1]), dtype=torch.long, device=inps.device
            )

            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (1,  batch[0].shape[1]),
                inps,
                0,
            )
        try:
            position_ids = cache["position_ids"]
        except:
            position_ids = None
        model.config.use_cache = use_cache

        return inps, outs, attention_mask, position_ids

    def _compute_task_loss(self, model, data):
        loss, outputs = task_loss(model, data)
        if loss is None:
            loss = torch.tensor(0.0, device=next(model.parameters()).device)
        return loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        target_data = self._get_batch_data(inputs, "target")
        loss, outputs = self._compute_task_loss(model, target_data)
        return (loss, outputs) if return_outputs else loss

    def compute_loss_adapted(self, model, inputs, key, CL=False, FT=False, return_outputs=False):
        data = self._get_batch_data(inputs, key)
        loss, outputs = self._compute_task_loss(model, data)
        if FT:
            try:
                pervasiveness_data = self._get_batch_data(inputs, "pervasiveness")
            except KeyError:
                pervasiveness_data = None
            if pervasiveness_data is not None:
                pervasiveness_loss, pervasiveness_outputs = self._compute_task_loss(
                    model, pervasiveness_data
                )
                loss = loss + pervasiveness_loss
                outputs = outputs if outputs is not None else pervasiveness_outputs
        return (loss, outputs) if return_outputs else loss
