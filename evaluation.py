from datasets import datasetSingleFrame, datasetVideoStackFrames, datasetVideoListFrames


def wrong_classification(model, testset):
    """
    Make a function that takes a model and shows some of the wrong predictions
    """
    pass


def saliency_map(model, testset):
    """
    Make a function that takes a model and shows some of the saliency maps of some chosen images of hot dogs
    """
    pass


def smooth_grad_saliency_map(model, testset):
    pass


if __name__ == '__main__':
    model = ...
    _, (trainset, testset) = dataset(batch_size=64)

    wrong_classification(model, testset)

    saliency_map(model, testset)

    smooth_grad_saliency_map(model, testset)
