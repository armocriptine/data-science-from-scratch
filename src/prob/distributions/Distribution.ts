import { Vector } from "../../vector/Vector";

export abstract class Distribution {
  public abstract sample(): number;
}
