import * as tfjs from '@tensorflow/tfjs-node';
import * as faceapi from 'face-api.js';
import * as fs from 'fs';
import * as cv from 'opencv';

const minConfidence = 0.5;
const faceDetectionNet = faceapi.nets.ssdMobilenetv1;
const faceDetectionOptions = new faceapi.SsdMobilenetv1Options({minConfidence});

let globalFrame = null;
let lastFrame = null;
function processFrame() {
  if (globalFrame == null) return;
  if (lastFrame == globalFrame) return;

  lastFrame = globalFrame;
  let tFrame = faceapi.tf.tensor3d(globalFrame, [480, 640, 3])
  faceapi.detectAllFaces(tFrame, faceDetectionOptions).then((faces) => {
    console.log(faces);
  });
}
setInterval(processFrame, 100);

async function run() {
  await faceDetectionNet.loadFromDisk('./weights');
  let video = '/dev/video0';
  let cap = new cv.VideoCapture(video);
  let capture = () => {
    cap.read(function(err, frame) {
      if (frame.width() > 0) {
        let data = new Uint8Array(frame.getData().buffer);
        globalFrame = data;
      }

      setTimeout(capture, 0);
    });
  }
  capture();
}

run()
