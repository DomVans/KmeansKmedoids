import React, { useState } from "react";

// Euclidean distance
const distance = (a, b) =>
  Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);

// KMeans clustering
function kMeans(points, k, maxIter = 100) {
  let centroids = points.slice(0, k).map((p) => [...p]);
  let assignments = new Array(points.length).fill(-1);
  const logs = [];

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    const prevCentroids = centroids.map((c) => [...c]);

    // Assign points
    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      let minDist = Infinity;
      let clusterIndex = -1;
      for (let c = 0; c < k; c++) {
        const dist = distance(point, centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          clusterIndex = c;
        }
      }
      if (assignments[i] !== clusterIndex) {
        assignments[i] = clusterIndex;
        changed = true;
      }
    }

    if (!changed) {
      // Log the final stable iteration
      logs.push({
        iter,
        type: "kmeans",
        initial: prevCentroids,
        updated: centroids.map((c) => [...c]),
      });
      break;
    }

    // Update centroids
    for (let c = 0; c < k; c++) {
      const assignedPoints = points.filter((_, i) => assignments[i] === c);
      if (assignedPoints.length === 0) continue;
      const meanX = assignedPoints.reduce((sum, p) => sum + p[0], 0) / assignedPoints.length;
      const meanY = assignedPoints.reduce((sum, p) => sum + p[1], 0) / assignedPoints.length;
      centroids[c] = [meanX, meanY];
    }

    logs.push({
      iter,
      type: "kmeans",
      initial: prevCentroids,
      updated: centroids.map((c) => [...c]),
    });
  }

  return { assignments, centroids, logs };
}

// KMedoids clustering
function kMedoids(points, k, maxIter = 100) {
  let medoids = points.slice(0, k).map((p) => [...p]);
  let assignments = new Array(points.length).fill(-1);
  const logs = [];

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;
    const prevMedoids = medoids.map((m) => [...m]);

    // Assign points
    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      let minDist = Infinity;
      let clusterIndex = -1;
      for (let m = 0; m < k; m++) {
        const dist = distance(point, medoids[m]);
        if (dist < minDist) {
          minDist = dist;
          clusterIndex = m;
        }
      }
      if (assignments[i] !== clusterIndex) {
        assignments[i] = clusterIndex;
        changed = true;
      }
    }

    if (!changed) {
      // Log the final stable iteration
      logs.push({
        iter,
        type: "kmedoids",
        initial: prevMedoids,
        updated: medoids.map((m) => [...m]),
      });
      break;
    }

    // Update medoids
    for (let m = 0; m < k; m++) {
      const assignedPoints = points.filter((_, i) => assignments[i] === m);
      if (assignedPoints.length === 0) continue;

      let bestMedoid = medoids[m];
      let bestCost = Infinity;

      for (const candidate of assignedPoints) {
        const cost = assignedPoints.reduce(
          (sum, p) => sum + distance(p, candidate),
          0
        );
        if (cost < bestCost) {
          bestCost = cost;
          bestMedoid = candidate;
        }
      }
      medoids[m] = [...bestMedoid];
    }

    logs.push({
      iter,
      type: "kmedoids",
      initial: prevMedoids,
      updated: medoids.map((m) => [...m]),
    });
  }

  return { assignments, medoids, logs };
}

const COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899"];

export default function Clustering() {
  const [points, setPoints] = useState([
    [30, 20],
    [40, 25],
    [45, 30],
    [90, 85],
    [85, 80],
    [88, 78],
    [50, 80],
    [55, 75],
    [53, 78],
  ]);
  const [k, setK] = useState(3);
  const [algorithm, setAlgorithm] = useState("kmeans");
  const [result, setResult] = useState(null);
  const [logs, setLogs] = useState([]);

  const width = 400;
  const height = 400;
  const padding = 40;

  const maxX = Math.max(...points.map((p) => p[0]));
  const maxY = Math.max(...points.map((p) => p[1]));
  const minX = Math.min(...points.map((p) => p[0]));
  const minY = Math.min(...points.map((p) => p[1]));

  const normalize = ([x, y]) => [
    padding + ((x - minX) / (maxX - minX || 1)) * (width - 2 * padding),
    height - (padding + ((y - minY) / (maxY - minY || 1)) * (height - 2 * padding)),
  ];

  const handleRun = () => {
    if (k <= 0 || k > points.length) {
      alert("Choose a valid number of clusters (k)");
      return;
    }
    let clusteringResult;
    if (algorithm === "kmeans") {
      clusteringResult = kMeans(points, k);
    } else {
      clusteringResult = kMedoids(points, k);
    }
    setResult(clusteringResult);
    setLogs(clusteringResult.logs || []);
  };

  const addPoint = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left - padding) / (width - 2 * padding)) * (maxX - minX) + minX;
    const y = maxY - ((e.clientY - rect.top - padding) / (height - 2 * padding)) * (maxY - minY);
    if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
      setPoints([...points, [x, y]]);
      setResult(null);
      setLogs([]);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-4 space-y-4">
      <h1 className="text-2xl font-bold">KMeans & KMedoids Clustering</h1>

      <div className="flex items-center space-x-4">
        <label>
          Algorithm:
          <select
            className="ml-2 border rounded p-1"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
          >
            <option value="kmeans">KMeans</option>
            <option value="kmedoids">KMedoids</option>
          </select>
        </label>

        <label>
          Clusters (k):
          <input
            type="number"
            min={1}
            max={points.length}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="ml-2 w-16 border rounded p-1"
          />
        </label>

        <button
          onClick={handleRun}
          className="bg-blue-600 text-white rounded px-4 py-2 hover:bg-blue-700"
        >
          Run
        </button>
      </div>

      <p className="text-sm text-gray-600">Click on the plot to add new points.</p>

      <svg
        onClick={addPoint}
        className="border rounded"
        width={width}
        height={height}
        style={{ backgroundColor: "#f0f0f0" }}
      >
        {/* Points */}
        {points.map(([x, y], i) => {
          const [nx, ny] = normalize([x, y]);
          const clusterIndex = result ? result.assignments[i] : -1;
          const color = clusterIndex >= 0 ? COLORS[clusterIndex % COLORS.length] : "#888";
          return (
            <circle
              key={i}
              cx={nx}
              cy={ny}
              r={6}
              fill={color}
              stroke="#333"
              strokeWidth={clusterIndex >= 0 ? 2 : 1}
            />
          );
        })}

        {/* Centroids or Medoids */}
        {result && (() => {
          const centers = algorithm === "kmeans" ? result.centroids : result.medoids;
          if (!Array.isArray(centers)) return null;
          return centers.map(([x, y], i) => {
            if (x === undefined || y === undefined) return null;
            const [nx, ny] = normalize([x, y]);
            return (
              <rect
                key={"center-" + i}
                x={nx - 8}
                y={ny - 8}
                width={16}
                height={16}
                fill={COLORS[i % COLORS.length]}
                stroke="#000"
                strokeWidth={3}
                rx={4}
                ry={4}
              />
            );
          });
        })()}
      </svg>

      {/* Iteration Logs */}
      {logs.length > 0 && (
        <div className="bg-white border rounded p-4 mt-4">
          <h2 className="font-semibold text-lg mb-2">Iteration Log</h2>
          {logs.map((log) => (
            <div key={log.iter} className="mb-4">
              <h3 className="font-medium">Iteration {log.iter + 1}</h3>
              <div className="text-sm">
                <p className="font-semibold mt-1">Initial {log.type === "kmeans" ? "Centroids" : "Medoids"}:</p>
                <ul className="ml-4 list-disc">
                  {log.initial.map((p, i) => (
                    <li key={i}>[{p[0].toFixed(2)}, {p[1].toFixed(2)}]</li>
                  ))}
                </ul>
                <p className="font-semibold mt-2">Updated {log.type === "kmeans" ? "Centroids" : "Medoids"}:</p>
                <ul className="ml-4 list-disc">
                  {log.updated.map((p, i) => (
                    <li key={i}>[{p[0].toFixed(2)}, {p[1].toFixed(2)}]</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}




