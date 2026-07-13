from .base import BaseFinetuningTrainer


class TargetFT(BaseFinetuningTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target_loss, outputs = self.target_loss(model, inputs)
        loss = self.target_weight * target_loss
        return (loss, outputs) if return_outputs else loss

    def iter_loss_terms(self, model, inputs):
        yield from self.iter_target_loss_terms(model, inputs)


class TargetFT_PervasivenessFT(BaseFinetuningTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target_loss, target_outputs = self.target_loss(model, inputs)
        pervasiveness_loss, pervasiveness_outputs = self.pervasiveness_ce_loss(model, inputs)
        loss = self.target_weight * target_loss + self.pervasiveness_weight * pervasiveness_loss
        outputs = target_outputs if target_outputs is not None else pervasiveness_outputs
        return (loss, outputs) if return_outputs else loss

    def iter_loss_terms(self, model, inputs):
        yield from self.iter_target_loss_terms(model, inputs)
        yield from self.iter_pervasiveness_ce_loss_terms(model, inputs)


class TargetFT_PervasivenessKL(BaseFinetuningTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        target_loss, target_outputs = self.target_loss(model, inputs)
        pervasiveness_loss, pervasiveness_outputs = self.pervasiveness_kl_loss(model, inputs)
        loss = self.target_weight * target_loss + self.kl_weight * pervasiveness_loss
        outputs = target_outputs if target_outputs is not None else pervasiveness_outputs
        return (loss, outputs) if return_outputs else loss

    def iter_loss_terms(self, model, inputs):
        yield from self.iter_target_loss_terms(model, inputs)
        yield from self.iter_pervasiveness_kl_loss_terms(model, inputs)


class TargetFT_L1(TargetFT_PervasivenessFT):
    def compute_loss(self, model, inputs, return_outputs=False):
        base_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss = base_loss + self.alpha * self.l1_loss(model)
        return (loss, outputs) if return_outputs else loss

    def iter_loss_terms(self, model, inputs):
        yield from super().iter_loss_terms(model, inputs)
        if self.alpha:
            yield "l1", self.alpha * self.l1_loss(model)
