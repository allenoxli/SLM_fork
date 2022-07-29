import torch


class BaseModelOuput:

    def __init__(self, **kwarg):

        self.logits = kwarg.get('logits')
        self.logits2 = kwarg.get('logits2')

        self.embeds = kwarg.get('embeds')
        self.decoder_hidden = kwarg.get('decoder_hidden')
        self.prev_hidden = kwarg.get('prev_hidden')
        self.cur_hidden = kwarg.get('cur_hidden')

        self.cell_state = kwarg.get('cell_state')
        self.input_gate = kwarg.get('input_gate')
        self.forget_gate = kwarg.get('forget_gate')

    def log_model_value(self, writer, step):
        self.log_value(writer, step, self.input_gate[0].squeeze(-1), 'input_gate')
        self.log_value(writer, step, self.forget_gate[0].squeeze(-1), 'forget_gate')
        self.log_value(writer, step, self.cell_state[0].sum(dim=-1), 'cell_state')
        self.log_value(writer, step, self.prev_hidden[0].sum(dim=-1), 'prev_hidden')
        self.log_value(writer, step, self.cur_hidden[0].sum(dim=-1), 'cur_hidden')
        self.log_value(writer, step, self.decoder_hidden[0].sum(dim=-1), 'decoder_hidden')

    @torch.no_grad()
    def log_value(self, writer, step, val, name):
        val = val.cpu()
        q_pos = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        q_val = torch.quantile(val, q_pos, dim=-1)
        # torch.quantile(val[0], q)
        writer.add_scalar(f'{name}/min', q_val[0].item(), step)
        writer.add_scalar(f'{name}/q1', q_val[1].item(), step)
        writer.add_scalar(f'{name}/q2', q_val[2].item(), step)
        writer.add_scalar(f'{name}/q3', q_val[3].item(), step)
        writer.add_scalar(f'{name}/max', q_val[4].item(), step)

        writer.add_scalar(f'{name}/mean', val.mean().item(), step)
        writer.add_scalar(f'{name}/std', val.std().item(), step)
        writer.add_scalar(f'{name}/sum', val.sum().item(), step)

class SegmentOutput(BaseModelOuput):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def log_model_value(self, writer, step):
        self.log_value(writer, step, self.decoder_hidden[0].sum(dim=-1), 'decoder_hidden')


class TextClassficationOutput:

    def __init__(self, **kwarg):

        self.gate_model_output = kwarg.get('gate_model_output')

        self.logits = kwarg.get('logits')
        self.hidden_state = kwarg.get('hidden_state')

        self.tkzer_hids = kwarg.get('tkzer_hids')
        self.tkzer_ids = kwarg.get('tkzer_ids')


class QAModelOuput:

    def __init__(self, **kwarg):

        self.logits = kwarg.get('logits')
        self.start_logits = kwarg.get('start_logits')
        self.end_logits = kwarg.get('end_logits')

        self.hidden_states = kwarg.get('hidden_states')
        self.tkzer_hids = kwarg.get('tkzer_hids')
        self.tkzer_ids = kwarg.get('tkzer_ids')


if __name__ == '__main__':

    a = BaseModelOuput(logit=[1, 2, 3], loss=12, cell_state=torch.ones(2, 3))

    print(a)
    print(a.logit)
    print(a.loss)
    print(a.cell_state)
    print(a.attn)
    print('----')
