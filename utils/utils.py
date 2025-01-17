import torch as th


def totorch(x, dtype=th.float32, device="cuda"):
    return th.as_tensor(x, dtype=dtype, device=device)


def tonumpy(x):
    return x.data.cpu().numpy()


def dim_check(tensor1, tensor2):
    assert (
        tensor1.shape == tensor2.shape
    ), f"Shapes are {tensor1.shape} vs {tensor2.shape}"
