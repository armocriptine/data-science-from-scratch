import { sum } from "../../utils/helpers";
import { Distribution } from "./Distribution";
import { UniformDistribution } from "./Uniform";

export class GaussianDistribution extends Distribution {
  private readonly uniform;

  constructor(public readonly mean: number, public readonly variance: number) {
    super();
    this.uniform = new UniformDistribution(0, 1);
  }

  public sample(): number {
    // Box-Muller transform
    const u1 = this.uniform.sample();
    const u2 = this.uniform.sample();

    return (
      Math.sqrt(-2 * Math.log(u1)) *
        Math.cos(2 * Math.PI * u2) *
        Math.sqrt(this.variance) +
      this.mean
    );
  }
}
