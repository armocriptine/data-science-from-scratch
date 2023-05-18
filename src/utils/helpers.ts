import fs from "fs";

export const sum = (prev: number, curr: number) => prev + curr;

export const square = (x: number) => x * x;

export const max = (prev: number, curr: number) => (prev > curr ? prev : curr);

export const clamp = (min: number, max: number) => (x: number) =>
  Math.max(Math.min(x, max), min);

export const maxIndex = (arr: readonly number[]) => {
  let index = 0;
  let max = Math.min(...arr);

  for (const [i, x] of arr.entries()) {
    if (x > max) {
      index = i;
      max = x;
    }
  }

  return index;
};

export const field =
  <T, K extends keyof T>(key: K) =>
  (entry: T): T[K] =>
    entry[key];

export function parseMnistDataSet(filePath: string): {
  features: number[];
  response: number;
}[] {
  const lines = fs
    .readFileSync(filePath)
    .toString()
    .trim()
    .split("\n")
    .map((line) => line.trim());

  return lines.slice(1).map((l) => {
    const cells = l.split(",").map((x) => parseInt(x));

    return {
      features: cells.slice(1),
      response: cells[0],
    };
  });
}

export const O = 1e-8;
export const I = 1 - O;