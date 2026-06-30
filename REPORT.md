# Screen Recapture Detection Report

## Approach

The objective of this assignment was to distinguish between genuine photographs and photographs of another electronic display.

My final solution combines a lightweight convolutional neural network with handcrafted frequency-domain information.

The image itself is processed using **MobileNetV3Small**, initialized with ImageNet pretrained weights. Rather than training a CNN from scratch on a small dataset, transfer learning allows the network to leverage robust low-level visual features while requiring only a small amount of task-specific training.

Alongside the CNN, I compute a handcrafted FFT-based moiré score. When a camera photographs another display, interference between the display's pixel grid and the camera sensor often creates periodic frequency artifacts. This handcrafted feature provides complementary information that the CNN may not always learn reliably from a small dataset.

The MobileNet feature vector and normalized moiré score are concatenated before the final classifier. During inference, lightweight Test-Time Augmentation (original image, horizontal flip and center crop) is applied and the predicted probabilities are averaged.

---

## Dataset

- REAL images: **72**
- SCREEN images: **70**

The dataset was collected manually using multiple lighting conditions, viewing angles and devices.

---

## Validation Performance

Best validation accuracy:

**86.21%**

Additional evaluation on the collected dataset:

- Accuracy: **78.87%**
- Precision: **75.64%**
- Recall: **84.29%**
- F1 Score: **79.73%**

These values are reported honestly and are based on my collected dataset. Final performance on unseen data will naturally depend on the diversity of the evaluation set.

---

## Latency

**Device**

Apple MacBook Air (Apple Silicon CPU)

**Measured Command-Line Runtime**

Running `predict.py` as a standalone script takes approximately **6–10 seconds per image**.

This measurement includes:

- Python interpreter startup
- TensorFlow import
- Model loading from disk
- Image preprocessing
- FFT feature computation
- Model inference

In a production deployment where the model is loaded once and reused for multiple predictions, only preprocessing and inference are performed for each image, resulting in significantly lower effective per-image latency.


---

## Cost per Image

The current implementation performs inference entirely on-device.

**Estimated cost:** approximately **$0 per image**, since no cloud infrastructure is required during inference.

If deployed on a cloud CPU for large-scale inference, the model is sufficiently lightweight that the expected compute cost would remain only a few cents per thousand images depending on the infrastructure and utilization assumptions.

---

## Trade-offs

The goal was to produce a lightweight solution suitable for mobile deployment rather than maximizing model complexity.

The hybrid approach combines deep learning with handcrafted computer vision features while maintaining relatively fast CPU inference and a small model size.

---

## Improvements with More Time

Given additional time I would:

- collect several thousand additional images across many devices and lighting conditions,
- introduce additional handcrafted frequency-domain descriptors,
- perform k-fold cross-validation,
- compress the model using TensorFlow Lite and quantization,
- continuously retrain using newly collected difficult examples,
- calibrate the operating threshold using production precision-recall curves instead of a single validation split.

---

## Adapting to Future Cheating Methods

As new cheating strategies emerge, I would continuously collect hard false-positive and false-negative examples, periodically retrain the model, and maintain a balanced dataset covering newer devices, displays and environmental conditions.

The decision threshold would be selected based on the desired balance between fraud detection rate and false alarm rate depending on the production use case.