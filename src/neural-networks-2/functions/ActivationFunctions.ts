import { square, sum } from "../../utils/helpers";
import { ActivationFunc } from "./Function";

export class Identity extends ActivationFunc<number> {
  public evaluate(x: number): number {
    return x;
  }

  public differentiate(): number {
    return 1;
  }
}

export class Sigmoid extends ActivationFunc<number> {
  public evaluate(input: number): number {
    return Math.max(Math.min(1 / (1 + Math.exp(-input)), 0.999), -0.999);
  }

  public differentiate(input: number): number {
    return this.evaluate(input) * (1 - this.evaluate(input));
  }
}

export class Tanh extends ActivationFunc<number> {
  public evaluate(input: number): number {
    return Math.tanh(input);
  }

  public differentiate(input: number): number {
    return 1 - this.evaluate(input) * this.evaluate(input);
  }
}

export class Relu extends ActivationFunc<number> {
  public evaluate(input: number): number {
    if (input > 0) {
      return input;
    } else {
      return 0;
    }
  }

  public differentiate(input: number): number {
    if (input > 0) {
      return 1;
    } else {
      return 0;
    }
  }
}

export class Lrelu extends ActivationFunc<number> {
  constructor(private readonly a: number) {
    super();
  }

  public evaluate(input: number): number {
    if (input) {
      return input;
    } else {
      return this.a * input;
    }
  }

  public differentiate(input: number): number {
    if (input > 0) {
      return 1;
    } else {
      return this.a;
    }
  }
}

export class Gelu extends ActivationFunc<number> {
  public evaluate(input: number): number {
    return input * this.approxGaussianCdf(input);
  }

  public differentiate(input: number): number {
    return this.approxGaussianCdf(input) + input * this.gaussianPdf(input);
  }

  private approxGaussianCdf(input: number): number {
    return (
      0.5 *
      (1 +
        Math.tanh(
          Math.sqrt(2 / Math.PI) * (input + 0.044715 * Math.pow(input, 3))
        ))
    );
  }

  private gaussianPdf(input: number): number {
    return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * square(input));
  }
}

export class SoftPlus extends ActivationFunc<number> {
  public evaluate(input: number): number {
    return Math.log(1 + Math.exp(input));
  }

  public differentiate(input: number): number {
    return 1 / (1 + Math.exp(-input));
  }
}

export class Sum extends ActivationFunc<number[]> {
  public evaluate(input: number[]): number {
    return input.reduce((sum, x) => sum + x, 0);
  }

  public differentiate(input: number[]): number[] {
    return new Array(input.length).fill(1);
  }
}

export type MultiplyFunctionInput = { left: number; right: number };

export class Multiply extends ActivationFunc<MultiplyFunctionInput> {
  public evaluate(inputs: MultiplyFunctionInput): number {
    return inputs.left * inputs.right;
  }

  public differentiate(inputs: MultiplyFunctionInput): MultiplyFunctionInput {
    if (!inputs) {
      console.log("!");
    }
    return {
      left: inputs.right,
      right: inputs.left,
    };
  }
}

export class Add extends ActivationFunc<{ a: number; b: number }> {
  public evaluate(inputs: { a: number; b: number }): number {
    return inputs.a + inputs.b;
  }

  public differentiate(): {
    a: number;
    b: number;
  } {
    return {
      a: 1,
      b: 1,
    };
  }
}

export type SoftmaxInput = { numerator: number; others: number[] };

export class Softmax extends ActivationFunc<SoftmaxInput> {
  public evaluate(inputs: SoftmaxInput): number {
    if (!inputs) {
      console.log("!");
    }
    return (
      Math.exp(inputs.numerator) /
      [inputs.numerator, ...inputs.others]
        .map((x) => Math.exp(x))
        .reduce(sum, 0)
    );
  }

  public differentiate(inputs: SoftmaxInput): SoftmaxInput {
    const o = this.evaluate(inputs);
    return {
      numerator: o * (1 - o),
      others: inputs.others.map(
        (d, i) =>
          -this.evaluate({
            numerator: d,
            others: [...inputs.others, inputs.numerator].filter(
              (_, j) => i !== j
            ),
          }) * o
      ),
    };
  }
}

export type NormalizeInput = {
  main: number;
  others: number[];
};

export class Normalize extends ActivationFunc<NormalizeInput> {
  constructor(public readonly e: number) {
    super();
  }

  public evaluate(input: NormalizeInput): number {
    const { mean, sd } = this.calculate(input);

    return (input.main - mean) / sd;
  }

  public differentiate(input: NormalizeInput): NormalizeInput {
    const { n, mean, sd } = this.calculate(input);

    return {
      main: (n - 1) / (n * sd) - square(input.main - mean) / (n * sd * sd * sd),
      others: input.others.map(
        (x) =>
          -(1 / (n * sd)) -
          ((input.main - mean) * (x - mean)) / (n * sd * sd * sd)
      ),
    };
  }

  private calculate(input: NormalizeInput): {
    n: number;
    mean: number;
    sd: number;
  } {
    const n = input.others.length + 1;
    const mean = [...input.others, input.main].reduce(sum, 0) / n;
    const sd = Math.sqrt(
      [...input.others, input.main]
        .map((x) => x - mean)
        .map(square)
        .reduce(sum, 0) /
        n +
        this.e
    );

    return { n, mean, sd };
  }
}
