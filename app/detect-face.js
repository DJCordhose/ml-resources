// see https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html
function detectHaarFace(img, faceCascade) {
    const msize = new cv.Size(0, 0);
    const newImg = img;
    const gray = new cv.Mat();
    cv.cvtColor(newImg, gray, cv.COLOR_RGBA2GRAY, 0);
    const faces = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

    let dst = undefined;
    if (faces.size() > 0) {

        // We want a square image
        let rect = undefined;
        const width = faces.get(0).width
        const height = faces.get(0).height;
        try {
            if (width > height) {
                let diff = width - height
                rect = new cv.Rect(faces.get(0).x, faces.get(0).y - Math.round(diff / 2), width, width);
            } else {
                let diff = height - width
                rect = new cv.Rect(faces.get(0).x - Math.round(diff / 2), faces.get(0).y, height, height);
            }
        } catch (e) {
            console.log("Face detection failed: " + e)
        }

        dst = newImg.roi(rect);
    }

    gray.delete();
    faces.delete();

    return dst;
}

async function loadDataFile(cvFilePath, url) {
    // see https://docs.opencv.org/master/utils.js
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    cv.FS_createDataFile("/", cvFilePath, data, true, false, false);
}