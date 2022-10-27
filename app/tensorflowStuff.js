
async function initializeModel () {
    // return await tf.loadGraphModel('./tfjs-model-simple-vgg/model.json');
    // return await tf.loadLayersModel('./tfjs-model-simple-vgg/model.json');
    return await tf.loadLayersModel('./tfjs-model-resent50/model.json');
}

function predict(model){
    const resizedImgElement = document.querySelector("#normalizedCanvas");
    const img = tf.browser.fromPixels(resizedImgElement, numChannels=1).expandDims();
    const predictionTensor = model.predict(img.cast('float32'))
    return tensor2Array(predictionTensor)
}

function tensor2Array(tensor){
    const values = tensor.dataSync();
    return Array.from(values);
}
