import { Distribution } from "../distributions/Distribution";
import { SampleSpace } from "./SampleSpace";

export class NumberSampleSpace extends SampleSpace<number> {
  public static from(dist: Distribution, count: number): NumberSampleSpace {
    return new NumberSampleSpace(
      new Array(count).fill(0).map(() => dist.sample())
    );
  }

  public pmf(value: number): number {
    return super.prob((x) => x === value);
  }

  public cdf(value: number): number {
    return super.prob((x) => x <= value);
  }
}
