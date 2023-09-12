async function loadModel() {
  return await tf.loadLayersModel("/model.json");
}

function getDigitsFromPrediction(prediction) {
  return prediction.map(el => max(el));
}

function max(array) {
  let max = 0;
  let maxIndex = 0
  for (let ii = 0; ii < 10; ii++)
    if (array[ii] > max) {
      maxIndex = ii;
      max = array[ii];
    }

  return maxIndex;
}

async function makePrediction(model, data) {
  const x = tf.tensor(data).reshape([-1, 28, 28, 1]);
  const y = await model.predict(x).array();
  return getDigitsFromPrediction(y);
}



function onInputImageChanged() {
  console.log("hello");
}


async function main() {
  const model = await loadModel();

  const context = document.getElementById("canvas").getContext("2d");
  const resultP = document.getElementById("result");
  const imageInput = document.getElementById("input-image");

  imageInput.addEventListener("change", async (event) => {
    context.clearRect(0, 0, 28, 28);

    if (event.target.value === '')
      return;

    const image = new Image();

    image.onload = async () => {
      context.drawImage(image, 0, 0, 28, 28);

      const imageData = context.getImageData(0, 0, 28, 28).data;

      const data = [];
      for (let ii = 0; ii < imageData.length; ii += 2)
      data.push((imageData[ii] + imageData[++ii] + imageData[++ii]) / 3 / 255);

      const prediction = await makePrediction(model, [data]);
      resultP.textContent = prediction[0];
    };

    image.src = window.URL.createObjectURL(event.target.files[0]);
  })
}

main();
