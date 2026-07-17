'use strict';

let openCv = null;
let readyPosted = false;
let readinessCallbackAttached = false;

self.Module = {
  onRuntimeInitialized() {
    resolveOpenCv();
  },
};

try {
  importScripts('./opencv.js');
  resolveOpenCv();
} catch (error) {
  postFailure(error);
}

function resolveOpenCv() {
  try {
    const candidate = self.cv;
    if (candidate && candidate.Mat) {
      markReady(candidate);
      return;
    }
    if (candidate && typeof candidate.then === 'function' && !readinessCallbackAttached) {
      readinessCallbackAttached = true;
      candidate.then((api) => markReady(api));
    }
    setTimeout(resolveOpenCv, 80);
  } catch (error) {
    postFailure(error);
  }
}

function markReady(api) {
  if (!api || !api.Mat) return;
  openCv = api;
  if (!readyPosted) {
    readyPosted = true;
    self.postMessage({ type: 'ready' });
  }
}

self.onmessage = (event) => {
  const message = event.data || {};
  if (message.type !== 'frame' || !openCv) return;
  try {
    self.postMessage({ type: 'result', result: detectDocument(message.buffer, message.width, message.height) });
  } catch (error) {
    self.postMessage({ type: 'frame-error', message: error && error.message ? error.message : 'Precision detection failed' });
  }
};

function detectDocument(buffer, width, height) {
  const cv = openCv;
  const allocations = [];
  const own = (value) => { allocations.push(value); return value; };

  try {
    const rgba = own(cv.matFromArray(height, width, cv.CV_8UC4, new Uint8ClampedArray(buffer)));
    const gray = own(new cv.Mat());
    const equalized = own(new cv.Mat());
    const blurred = own(new cv.Mat());
    const equalizedBlur = own(new cv.Mat());
    const normalEdges = own(new cv.Mat());
    const lowContrastEdges = own(new cv.Mat());
    const combined = own(new cv.Mat());
    const closed = own(new cv.Mat());
    const contours = own(new cv.MatVector());
    const hierarchy = own(new cv.Mat());
    const kernel = own(cv.Mat.ones(5, 5, cv.CV_8U));

    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    cv.equalizeHist(gray, equalized);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.GaussianBlur(equalized, equalizedBlur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.Canny(blurred, normalEdges, 42, 128, 3, true);
    cv.Canny(equalizedBlur, lowContrastEdges, 24, 82, 3, true);
    cv.bitwise_or(normalEdges, lowContrastEdges, combined);
    cv.morphologyEx(combined, closed, cv.MORPH_CLOSE, kernel, new cv.Point(-1, -1), 2);
    cv.dilate(closed, closed, kernel, new cv.Point(-1, -1), 1);
    cv.findContours(closed, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const frameArea = width * height;
    let best = null;
    for (let index = 0; index < contours.size(); index += 1) {
      const contour = contours.get(index);
      try {
        const area = Math.abs(cv.contourArea(contour));
        if (area < frameArea * .12 || area > frameArea * .985) continue;
        const perimeter = cv.arcLength(contour, true);
        let foundFourCornerPolygon = false;
        for (const epsilon of [.012, .02, .03, .045, .065, .085]) {
          const polygon = new cv.Mat();
          try {
            cv.approxPolyDP(contour, polygon, epsilon * perimeter, true);
            if (polygon.rows !== 4 || !cv.isContourConvex(polygon)) continue;
            const points = polygonPoints(polygon);
            const ordered = orderCorners(points);
            const quadArea = polygonArea(ordered);
            if (quadArea < frameArea * .12) continue;
            const rectangularity = quadrilateralQuality(ordered);
            if (rectangularity < .36) continue;
            const score = quadArea * (.65 + rectangularity * .35);
            if (!best || score > best.score) best = { points: ordered, score };
            foundFourCornerPolygon = true;
          } finally {
            polygon.delete();
          }
        }

        if (!foundFourCornerPolygon) {
          const irregularQuad = contourExtremeQuad(contour);
          if (irregularQuad) {
            const quadArea = polygonArea(irregularQuad);
            const quality = quadrilateralQuality(irregularQuad);
            if (quadArea >= frameArea * .14 && quality >= .3) {
              const score = quadArea * (.44 + quality * .24);
              if (!best || score > best.score) best = { points: irregularQuad, score };
            }
          }
        }
      } finally {
        contour.delete();
      }
    }

    if (!best) return null;
    const normalized = best.points.map((point) => ({ x: point.x / width, y: point.y / height }));
    const margin = .018;
    const inside = normalized.every((point) => point.x > margin && point.x < 1 - margin && point.y > margin && point.y < 1 - margin);
    const brightness = cv.mean(gray)[0];
    const laplacian = own(new cv.Mat());
    const mean = own(new cv.Mat());
    const standardDeviation = own(new cv.Mat());
    cv.Laplacian(gray, laplacian, cv.CV_64F);
    cv.meanStdDev(laplacian, mean, standardDeviation);
    const sharpness = standardDeviation.doubleAt(0, 0) ** 2;

    return {
      quad: normalized,
      inside,
      brightness,
      brightnessOk: brightness >= 36 && brightness <= 252,
      sharpness,
      sharpEnough: sharpness >= 26,
    };
  } finally {
    for (let index = allocations.length - 1; index >= 0; index -= 1) {
      try { allocations[index].delete(); } catch (_) {}
    }
  }
}

function polygonPoints(polygon) {
  const points = [];
  for (let row = 0; row < polygon.rows; row += 1) {
    points.push({ x: polygon.intPtr(row, 0)[0], y: polygon.intPtr(row, 0)[1] });
  }
  return points;
}

function contourExtremeQuad(contour) {
  let topLeft = null;
  let topRight = null;
  let bottomRight = null;
  let bottomLeft = null;
  for (let row = 0; row < contour.rows; row += 1) {
    const values = contour.intPtr(row, 0);
    const point = { x: values[0], y: values[1] };
    const sum = point.x + point.y;
    const difference = point.x - point.y;
    if (!topLeft || sum < topLeft.score) topLeft = { ...point, score: sum };
    if (!topRight || difference > topRight.score) topRight = { ...point, score: difference };
    if (!bottomRight || sum > bottomRight.score) bottomRight = { ...point, score: sum };
    if (!bottomLeft || difference < bottomLeft.score) bottomLeft = { ...point, score: difference };
  }
  const points = [topLeft, topRight, bottomRight, bottomLeft];
  if (points.some((point) => !point)) return null;
  const unique = new Set(points.map((point) => Math.round(point.x) + ':' + Math.round(point.y)));
  if (unique.size !== 4) return null;
  return points.map(({ x, y }) => ({ x, y }));
}

function orderCorners(points) {
  const bySum = [...points].sort((a, b) => (a.x + a.y) - (b.x + b.y));
  const byDifference = [...points].sort((a, b) => (a.x - a.y) - (b.x - b.y));
  return [bySum[0], byDifference[3], bySum[3], byDifference[0]];
}

function polygonArea(points) {
  let total = 0;
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    total += point.x * next.y - next.x * point.y;
  });
  return Math.abs(total) / 2;
}

function quadrilateralQuality(points) {
  const lengths = points.map((point, index) => distance(point, points[(index + 1) % points.length]));
  const shortest = Math.min(...lengths);
  const longest = Math.max(...lengths);
  if (!shortest || longest / shortest > 7) return 0;
  let angleScore = 0;
  points.forEach((point, index) => {
    const previous = points[(index + 3) % points.length];
    const next = points[(index + 1) % points.length];
    const first = { x: previous.x - point.x, y: previous.y - point.y };
    const second = { x: next.x - point.x, y: next.y - point.y };
    const cosine = Math.abs((first.x * second.x + first.y * second.y) / Math.max(1, distance({ x: 0, y: 0 }, first) * distance({ x: 0, y: 0 }, second)));
    angleScore += Math.max(0, 1 - cosine);
  });
  return angleScore / 4;
}

function distance(first, second) {
  return Math.hypot(second.x - first.x, second.y - first.y);
}

function postFailure(error) {
  self.postMessage({ type: 'error', message: error && error.message ? error.message : 'Precision detector could not load' });
}
