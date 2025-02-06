import os

from qwen_vl_utils import process_vision_info
from sklearn.metrics import confusion_matrix
from torchvision.datasets import MNIST
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VL


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_image():
    test_set = MNIST(root='./data', train=False, download=True)
    yield from test_set


def inference(messages, processor, model):
    # process
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text],
                       images=image_inputs,
                       videos=None,
                       return_tensors='pt').to('cuda:0')
    # generate
    generated_ids = model.generate(**inputs, max_new_tokens=4)
    generated_ids = generated_ids[0, len(inputs.input_ids[0]):]
    # decode
    decoded = processor.decode(generated_ids,
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
    return decoded


def main():
    # create model and processor
    model = Qwen2_5_VL.from_pretrained('./ckpts/Qwen2.5-VL-3B-Instruct',
                                       torch_dtype='auto',
                                       device_map='cuda:0')
    processor = AutoProcessor.from_pretrained('./ckpts/Qwen2.5-VL-3B-Instruct')
    # prepare data
    messages = [
        {
            'role':
            'user',
            'content': [{
                'type': 'image',
                'image': None
            }, {
                'type':
                'text',
                'text':
                'What is the digit (0-9)? Response with a single digit.'
            }]
        },
    ]
    accuracy = AverageMeter()
    labels, results = [], []
    # test
    pbar = tqdm(total=10000)
    for image, label in get_image():
        messages[0]['content'][0]['image'] = image
        result = inference(messages, processor, model)
        if int(result) == label:
            accuracy.update(1)
        else:
            accuracy.update(0)
        labels.append(label)
        results.append(int(result))
        pbar.set_description(f'Accuracy: {100 * accuracy.avg:.2f}%')
        pbar.update(1)
    # report
    pbar.close()
    this_dir = os.path.dirname(__file__)
    with open(f'{this_dir}/qwen2_5_vl_3b_accuracy.txt', 'w') as f:
        f.write(f'Accuracy: {100 * accuracy.avg:.2f}%')
        # write confusion matrix
        f.write('\nConfusion matrix:\n')
        f.write(str(confusion_matrix(labels, results)))
    with open(f'{this_dir}/qwen2_5_vl_3b_response.txt', 'w') as f:
        for label, result in zip(labels, results):
            f.write(f'Label: {label}, Result: {result}\n')


if __name__ == '__main__':
    main()
