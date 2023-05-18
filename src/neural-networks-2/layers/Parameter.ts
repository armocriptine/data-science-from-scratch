import { Vector } from "../../vector/Vector";
import { ParameterNode } from "../nodes/Parameter";
import { SimpleLayer } from "./Layer";

export class ParameterLayer extends SimpleLayer {
  constructor(size: number, initialValues: Vector) {
    super(
      Array.from({ length: size }).map(
        (_, i) => new ParameterNode(initialValues.at(i))
      )
    );
  }
}
