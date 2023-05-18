import { Distribution } from "./Distribution";

export class UniformDistribution extends Distribution {
  constructor(public readonly min: number, public readonly max: number) {
    super();
  }

  public sample(): number {
    return Math.random() * (this.max - this.min) + this.min;
  }
}
