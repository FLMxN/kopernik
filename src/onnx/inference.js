import * as ort from 'onnxruntime-web';

import {
  COUNTRY_CODES,
  COUNTRY_NAMES,
  REGION_CODES,
  REGION_EMOJI,
  REGION_NAMES,
  UNDEFINED_CODES,
} from './labels.js';

const PRECISION = import.meta.env.VITE_ONNX_PRECISION || 'fp16';
const WIDTH = 997;
const HEIGHT = 561;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let sessionsPromise = null;

export function flagEmoji(code) {
  if (!/^[A-Z]{2}$/.test(code)) return '🏳️';
  return String.fromCodePoint(
    ...[...code].map((char) => 0x1f1e6 + char.charCodeAt(0) - 65),
  );
}

async function hasWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function createSessions(onStatus) {
  const modelBase = `${import.meta.env.BASE_URL}models`;
  const webgpu = await hasWebGPU();
  const options = {
    executionProviders: webgpu ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
  };

  onStatus?.('Loading ONNX models (first run may download ~90 MB)...');
  let provider = webgpu ? 'webgpu' : 'wasm';
  let country;
  let region;

  try {
    country = await ort.InferenceSession.create(
      `${modelBase}/country_${PRECISION}.onnx`,
      options,
    );
    region = await ort.InferenceSession.create(
      `${modelBase}/region_${PRECISION}.onnx`,
      options,
    );
  } catch (error) {
    console.warn('WebGPU session failed, falling back to WASM', error);
    provider = 'wasm';
    country = await ort.InferenceSession.create(
      `${modelBase}/country_${PRECISION}.onnx`,
      { ...options, executionProviders: ['wasm'] },
    );
    region = await ort.InferenceSession.create(
      `${modelBase}/region_${PRECISION}.onnx`,
      { ...options, executionProviders: ['wasm'] },
    );
  }

  return { country, region, provider };
}

export function loadSessions(onStatus) {
  if (!sessionsPromise) {
    sessionsPromise = createSessions(onStatus).catch((error) => {
      sessionsPromise = null;
      throw error;
    });
  }
  return sessionsPromise;
}

async function imageToTensor(file) {
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, {
      resizeWidth: WIDTH,
      resizeHeight: HEIGHT,
      resizeQuality: 'low',
    });
  } catch {
    bitmap = await createImageBitmap(file);
  }
  const canvas = document.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, WIDTH, HEIGHT);
  bitmap.close?.();

  const { data } = context.getImageData(0, 0, WIDTH, HEIGHT);
  const plane = WIDTH * HEIGHT;
  const input = new Float32Array(3 * plane);

  for (let index = 0; index < plane; index += 1) {
    const offset = index * 4;
    input[index] = data[offset] / 255;
    input[plane + index] = data[offset + 1] / 255;
    input[2 * plane + index] = data[offset + 2] / 255;
  }

  for (let channel = 0; channel < 3; channel += 1) {
    const start = channel * plane;
    for (let index = 0; index < plane; index += 1) {
      input[start + index] = (input[start + index] - MEAN[channel]) / STD[channel];
    }
  }

  return new ort.Tensor('float32', input, [1, 3, HEIGHT, WIDTH]);
}

function softmax(values) {
  const output = new Float32Array(values.length);
  let max = -Infinity;
  for (const value of values) {
    if (value > max) max = value;
  }
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.exp(values[index] - max);
    output[index] = value;
    sum += value;
  }
  for (let index = 0; index < output.length; index += 1) {
    output[index] /= sum;
  }
  return output;
}

function countryResults(probabilities, pretty) {
  const items = COUNTRY_CODES.map((code, index) => ({
    code,
    name: COUNTRY_NAMES[code],
    prob: probabilities[index],
    flag: flagEmoji(code),
  }));

  if (pretty) {
    let undefinedProbability = 0;
    for (const code of UNDEFINED_CODES) {
      const index = COUNTRY_CODES.indexOf(code);
      if (index >= 0) {
        undefinedProbability += probabilities[index];
        items[index].prob = 0;
      }
    }
    if (undefinedProbability > 0) {
      items.push({
        code: 'UNDEFINED',
        name: 'Undefined',
        prob: undefinedProbability,
        flag: '🏳️',
      });
    }
  }

  return items
    .filter((item) => item.prob > 0)
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 8);
}

function regionResults(probabilities) {
  return REGION_CODES.map((code, index) => ({
    code,
    name: REGION_NAMES[code],
    prob: probabilities[index],
    emoji: REGION_EMOJI[code] || '🌍',
  }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);
}

export async function predictLocal(file, pretty, onStatus) {
  const { country: countrySession, region: regionSession, provider } =
    await loadSessions(onStatus);

  onStatus?.('Preprocessing image...');
  const tensor = await imageToTensor(file);

  onStatus?.('Running inference...');
  const started = performance.now();
  const countryOutput = await countrySession.run({
    [countrySession.inputNames[0]]: tensor,
  });
  const regionOutput = await regionSession.run({
    [regionSession.inputNames[0]]: tensor,
  });
  const elapsedMs = performance.now() - started;

  const countryProbabilities = softmax(
    countryOutput[countrySession.outputNames[0]].data,
  );
  const regionProbabilities = softmax(
    regionOutput[regionSession.outputNames[0]].data,
  );

  return {
    country: countryResults(countryProbabilities, Boolean(pretty)),
    region: regionResults(regionProbabilities),
    provider,
    elapsedMs,
  };
}
