import { Matrix } from "../matrix/Matrix";
import { I, O, maxIndex } from "../utils/helpers";
import { Vector } from "../vector/Vector";

export const embedThaiCharacters = (
  text: string,
  withPosition = true,
  padding?: number,
): Matrix => {
  if (padding) {
    text = text.padEnd(padding, '$');
  }

  const embedding = Array.from(text).map(
    (char, i) =>
      new Vector([
        ...embedBinary(getThaiCharIndex(char), 8),
        ...(withPosition ? embedBinary(i, 3) : []),
      ])
  );
  return Matrix.fromRows(embedding);
};

export const unembedThaiCharacters = (embeddings: Matrix): string => {
  return embeddings.rows
    .map((row) => getThaiCharByIndex(unembedBinary(row.entries)))
    .join("");
};

export const embedLatinCharacters = (
  text: string,
  withBeginning = true,
  withPosition = true,
  padding?: number,
  binary = true
): Matrix => {
  text = `${withBeginning ? "^" : ""}${text}`;
  if (padding) {
    text = text.padEnd(padding, '$');
  }

  const embedding = Array.from(text).map(
    (char, i) =>
      new Vector([
        ...(binary
          ? embedBinary(getLatinCharIndex(char), 8)
          : embedOneHot(getLatinCharIndex(char), 30)),
        ...(withPosition ? embedBinary(i, 3) : []),
      ])
  );

  return Matrix.fromRows(embedding);
};

export const unembedLatinCharacters = (embeddings: Matrix, binary = true): string => {
  return embeddings.rows
    .map((row) => getLatinCharByIndex(binary ? unembedBinary(row.entries) : unembedOneHot(row.entries)))
    .join("");
};

export const embedBinary = (num: number, digits: number): number[] => {
  let divided = num;
  const binary = new Array<number>(digits).fill(O);

  let i = 0;
  while (divided > 0) {
    const remainder = divided % 2;
    binary[i] = remainder === 1 ? I : O;
    divided = Math.floor(divided / 2);
    i++;
  }

  return binary;
};

export const embedOneHot = (num: number, max: number): number[] => {
  return Array.from({ length: max }).map((_, i) => (num === i ? I : O));
};

export const unembedOneHot = (vec: readonly number[]): number => {
  return maxIndex(vec);
};

export const unembedBinary = (bin: readonly number[]): number => {
  let sum = 0;
  let pow = 1;
  for (const x of bin) {
    sum += pow * (x >= 0.5 ? 1 : 0);
    pow *= 2;
  }
  return sum;
};

const getThaiCharIndex = (char: string): number => {
  if (char === " ") return 71;
  if (char === ".") return 72;
  if (char === "$") return 73;

  let charIndex = char.charCodeAt(0) - 3585;

  if (charIndex >= 63) {
    charIndex -= 5;
  }

  return charIndex;
};

export const getThaiCharByIndex = (index: number): string => {
  if (index === 71) return " ";
  if (index === 72) return ".";
  if (index === 73) return "$";
  return String.fromCharCode((index >= 58 ? index + 5 : index) + 3585);
};

const getLatinCharIndex = (char: string): number => {
  if (char === "^") return 0;
  if (char === "-") return 27;
  if (char === " ") return 28;
  if (char === "$") return 29;

  const charIndex = char.charCodeAt(0) - 97;

  return charIndex + 1;
};

export const getLatinCharByIndex = (index: number): string => {
  if (index === 0) return "^";
  if (index === 27) return "-";
  if (index === 28) return " ";
  if (index === 29) return "$";
  return String.fromCharCode(index + 97 - 1);
};
