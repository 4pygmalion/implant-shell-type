import torch
import tqdm
from .base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    """Gradient independent한 Score-CAM을 구함

    Args


    Note:
        1. 입력이미지(X)을 forwarding하여 Target layer에서의 Activation을 구함
        2. Activation의 채널마다 아래를 진행
         - 한 Activation channel을 원본이미지에 맞춰 업샘플링
         - 위에서 구한 activation을 채널에 한해서, Min-max scaling을 진행하여 한 채널에서의 정규화를진행
          (이 정규화 후에는 가장낮은값은 0으로, 가장 높은값은 1로 만들어짐)
         - 원본이미지와 정규화된 Activation과 element wise multiplication 진행(=Hadamard product), 각RGB에 다 곱함
         - 곱연산한 이미지를 모아놓음
        3. 모은 이미지를 Stack하여 이미지로구함.
        4. 원하는 클레스에서의 로짓을 구하고, 입력이미지와의 차이를 구함
        5. activation channel 기준으로 softmax을 구해 weight (a_^{c}_{k})을 구함
        6. Activations과 weight을 구한후 ReLU 연산을 통해 Heatmap을 구함

    Example:
        >>> from matplotlib import pyplot as plt
        >>> import numpy as np

        >>> fig, axes = plt.subplots(1, 9, figsize=(20, 10))
        >>> for i in range(1, 10):
        >>>     flattened_matrix = output.view(-1)
        >>>     num_elements = flattened_matrix.numel()
        >>>     num_lower_10_percent = int(0.1 *i* num_elements)
        >>>     _, indices = torch.topk(flattened_matrix, num_lower_10_percent, largest=False)

        >>>     mask_indices = np.unravel_index(indices.cpu().numpy(), (512,512))
        >>>     x, y = mask_indices
        >>>     copy_image = patchset.test[idx].image_array.copy()
        >>>     copy_image[x, y, :] = 0
        >>>     new_x = transform(copy_image).unsqueeze(0).to(device)
        >>>     confidence = torch.sigmoid(model(new_x)).item()
        >>>     axes[i - 1].imshow(copy_image)
        >>>     axes[i - 1].set_title("Score:"+f'{confidence:.3f}')


    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        device: str,
        verbose: bool = True,
    ):
        super(ScoreCAM, self).__init__(model, target_layer, device)
        self.verbose = verbose

    def _forward(self, x, class_idx: int) -> torch.Tensor:
        b, c, h, w = x.size()

        # 1. 원본이미지의 Logit을 구함 + hook으로 activation고 구해짐
        with torch.no_grad():
            logits = self.model(x)

        # 2. 관심클레스의 logit-> model confidence로 변경
        probs = torch.nn.functional.softmax(logits, dim=1)  # (N, class)
        probs = probs[0, class_idx]  # torch.Size([])

        # 3. Activation을 채널별로 조작하기위해 미리 저장
        activations = self.activations
        b, k, u, v = activations.size()

        # 4. Score-CAM을 위해 아래 진행하는 과정 진행
        score_saliency_map = torch.zeros((1, 1, w, h)).to(self.device)
        with torch.no_grad():
            if self.verbose:
                channel_iter = tqdm.tqdm(range(k))
            else:
                channel_iter = range(k)

            for i in channel_iter:
                # 각 Activation을 shape를 보존하여 얻음
                channel_activation = activations[:, [i], :, :]  #  (1, 1, 512, 512)

                # 업샘플링
                channel_activation = torch.nn.functional.interpolate(
                    input=channel_activation,
                    size=(h, w),
                    mode="bilinear",
                )

                # Min, Max가 같은 경우는 activation이 없으니 패스
                if channel_activation.max() == channel_activation.min():
                    continue

                # Min, Max scale
                norm_channel_activation = (
                    channel_activation - channel_activation.min()
                ) / (channel_activation.max() - channel_activation.min())

                # Elementwise multiplication
                output = self.model(x * norm_channel_activation)
                output = torch.nn.functional.softmax(output, dim=1)
                score = output[0][class_idx]
                score_saliency_map += score * channel_activation

        score_saliency_map = torch.nn.functional.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = (
            score_saliency_map.min(),
            score_saliency_map.max(),
        )

        score_saliency_map = (
            (score_saliency_map - score_saliency_map_min)
            .div(score_saliency_map_max - score_saliency_map_min)
            .data
        )

        return score_saliency_map

    def __call__(self, x, class_index: int):
        return self._forward(x, class_index)
