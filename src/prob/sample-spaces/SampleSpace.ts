import { DataSet } from "../../data/DataSet";
import { Distribution } from "../distributions/Distribution";
import { NumberSampleSpace } from "./NumberSampleSpace";

export class SampleSpace<T> {
  constructor(private readonly entries: readonly T[]) {}

  public prob(
    predicate: (x: T) => boolean,
    cond: (x: T) => boolean = () => true
  ) {
    return (
      this.entries.filter(cond).filter(predicate).length /
      this.entries.filter(cond).length
    );
  }

  public sampleByIndex(dist: Distribution): T {
    return this.entries[Math.round(dist.sample())];
  }
}
