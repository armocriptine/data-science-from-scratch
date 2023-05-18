import { Vector } from "../../vector/Vector";
import { Preactivations } from "../types/Preactivations";

export abstract class ActivationFunc<T = any> {
  public abstract evaluate(inputs: T): number;

  public abstract differentiate(inputs: T): T;
}
