import { Node } from "../nodes/Node";

export abstract class Layer {}

export class SimpleLayer<T extends Node = Node> extends Layer {
  constructor(public readonly outputNodes: T[]) {
    super();
  }

  public get outputSize(): number {
    return this.outputNodes.length;
  }
}
