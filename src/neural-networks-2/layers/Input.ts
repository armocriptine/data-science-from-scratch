import { InputNode } from "../nodes/Input";
import { SimpleLayer } from "./Layer";

export class InputLayer extends SimpleLayer<InputNode> {
  constructor(inputSize: number) {
    super(Array.from({ length: inputSize }).map(() => new InputNode()));
  }
}
