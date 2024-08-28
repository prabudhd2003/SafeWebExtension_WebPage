document.getElementById('imageInput').addEventListener('change', async function(event) {
    const file = event.target.files[0];
    if (file) {
        const imageUrl = URL.createObjectURL(file);
        const uploadedImage = document.getElementById('uploadedImage');
        const resultElement = document.getElementById('uploadResult');
        const viewUploadedButton = document.getElementById('viewUploadedImage');

        uploadedImage.src = imageUrl;
        resultElement.textContent = "Classifying...";

        const safetyStatus = await classifyImage(file);

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

async function classifyImage(imageFile) {
    const modelPath = 'image_safety_model.onnx';
    const session = await ort.InferenceSession.create(modelPath);
    
    const reader = new FileReader();
    reader.readAsDataURL(imageFile);

    return new Promise((resolve) => {
        reader.onload = async function(event) {
            const image = new Image();
            image.src = event.target.result;

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
        };
    });
}

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

function argMax(array) {
    return array.indexOf(Math.max(...array));
}

// Automatically classify the example images
window.onload = async function() {
    const safeExample = document.getElementById('safeExample');
    const unsafeExample = document.getElementById('unsafeExample');
    const viewAnywayButton = document.getElementById('viewAnyway');

    const safeStatus = await classifyImageFromUrl(safeExample.src);
    const unsafeStatus = await classifyImageFromUrl(unsafeExample.src);

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

async function classifyImageFromUrl(imageUrl) {
    const modelPath = 'image_safety_model.onnx';
    const session = await ort.InferenceSession.create(modelPath);

    return new Promise((resolve) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.src = imageUrl;

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
