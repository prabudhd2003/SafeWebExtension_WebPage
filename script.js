// Image upload functionality
document.getElementById('imageInput').addEventListener('change', async function(event) {
    const file = event.target.files[0];
    if (file) {
        const imageUrl = URL.createObjectURL(file);
        const uploadedImage = document.getElementById('uploadedImage');
        const resultElement = document.getElementById('uploadResult');
        const viewUploadedButton = document.getElementById('viewUploadedImage');

        resultElement.textContent = "Classifying...";

        const resizedImage = await resizeImage(imageUrl, 300, 300);
        uploadedImage.src = resizedImage;

        const safetyStatus = await classifyImageFromDataURL(resizedImage);

        resultElement.textContent = `Safety: ${safetyStatus}`;

        if (safetyStatus === 'unsafe') {
            uploadedImage.classList.add('blurred');
            viewUploadedButton.classList.remove('hidden');
        } else {
            uploadedImage.classList.remove('blurred');
            viewUploadedButton.classList.add('hidden');
        }

        viewUploadedButton.addEventListener('click', function() {
            uploadedImage.classList.remove('blurred');
            viewUploadedButton.classList.add('hidden');
        });
    }
});

// Image URL fetching functionality
document.getElementById('fetchImage').addEventListener('click', async function() {
    const url = document.getElementById('urlInput').value;
    if (url) {
        const urlImage = document.getElementById('urlImage');
        const urlResult = document.getElementById('urlResult');
        const viewUrlButton = document.getElementById('viewUrlImage');

        urlResult.textContent = "Fetching and Classifying...";

        const resizedImage = await resizeImage(url, 300, 300);
        urlImage.src = resizedImage;

        const safetyStatus = await classifyImageFromDataURL(resizedImage);

        urlResult.textContent = `Safety: ${safetyStatus}`;

        if (safetyStatus === 'unsafe') {
            urlImage.classList.add('blurred');
            viewUrlButton.classList.remove('hidden');
        } else {
            urlImage.classList.remove('blurred');
            viewUrlButton.classList.add('hidden');
        }

        viewUrlButton.addEventListener('click', function() {
            urlImage.classList.remove('blurred');
            viewUrlButton.classList.add('hidden');
        });
    }
});

// Resize image function
async function resizeImage(imageSrc, targetWidth, targetHeight) {
    return new Promise((resolve) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.src = imageSrc;

        image.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            const context = canvas.getContext('2d');
            context.drawImage(image, 0, 0, targetWidth, targetHeight);

            resolve(canvas.toDataURL());  // Return the resized image as a data URL
        };
    });
}

// Classify image from data URL
async function classifyImageFromDataURL(dataURL) {
    const modelPath = 'image_safety_model.onnx';
    const session = await ort.InferenceSession.create(modelPath);

    return new Promise((resolve) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.src = dataURL;

        image.onload = async function() {
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            const context = canvas.getContext('2d');
            context.drawImage(image, 0, 0, 224, 224);
            const imageData = context.getImageData(0, 0, 224, 224);
            const input = preprocessImage(imageData);

            const feeds = {};
            feeds[session.inputNames[0]] = new ort.Tensor('float32', input, [1, 3, 224, 224]);

            const outputData = await session.run(feeds);
            const output = outputData[session.outputNames[0]].data;

            const predictedIdx = argMax(output);
            const indexToCategory = {
                0: 'safe',
                1: 'unsafe'
            };

            resolve(indexToCategory[predictedIdx]);
        };
    });
}

// Preprocess image function
function preprocessImage(imageData) {
    const { data, width, height } = imageData;
    const float32Data = new Float32Array(width * height * 3);

    for (let i = 0; i < width * height; i++) {
        const r = data[i * 4] / 255;
        const g = data[i * 4 + 1] / 255;
        const b = data[i * 4 + 2] / 255;

        float32Data[i] = (r - 0.485) / 0.229;
        float32Data[i + width * height] = (g - 0.456) / 0.224;
        float32Data[i + width * height * 2] = (b - 0.406) / 0.225;
    }

    return float32Data;
}

// Utility function to find the index of the maximum value in an array
function argMax(array) {
    return array.indexOf(Math.max(...array));
}

// Automatically classify the example images
window.onload = async function() {
    const safeExample = document.getElementById('safeExample');
    const unsafeExample = document.getElementById('unsafeExample');
    const viewAnywayButton = document.getElementById('viewAnyway');

    const safeStatus = await classifyImageFromDataURL(await resizeImage(safeExample.src, 300, 300));
    const unsafeStatus = await classifyImageFromDataURL(await resizeImage(unsafeExample.src, 300, 300));

    if (safeStatus === 'safe') {
        document.querySelectorAll('.example .label')[0].textContent = 'Safe';
    }

    if (unsafeStatus === 'unsafe') {
        unsafeExample.classList.add('blurred');
        document.querySelectorAll('.example .label')[1].textContent = 'Unsafe';
        viewAnywayButton.classList.remove('hidden');

        viewAnywayButton.addEventListener('click', function() {
            unsafeExample.classList.remove('blurred');
            viewAnywayButton.classList.add('hidden');
        });
    }
};
