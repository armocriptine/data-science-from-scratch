import { Vector } from "../../vector/Vector";
import { AddNode } from "../nodes/Add";
import { MultiplyNode } from "../nodes/Multiply";
import { NormNode } from "../nodes/Norm";
import { SimpleLayer } from "./Layer";
import { ParameterLayer } from "./Parameter";

export class NormLayer extends SimpleLayer {
  constructor(incomingLayer: SimpleLayer) {
    const prevNodes = incomingLayer.outputNodes;

    const normLayer = new SimpleLayer(
      prevNodes.map(
        (node, i) =>
          new NormNode(
            node,
            prevNodes.filter((_, j) => i !== j),
            1e-8
          )
      )
    );

    const scaleParams = new ParameterLayer(
      normLayer.outputNodes.length,
      Vector.repeat(1, normLayer.outputNodes.length)
    );

    const scaleLayer = new SimpleLayer(
      normLayer.outputNodes.map(
        (node, i) => new MultiplyNode(scaleParams.outputNodes[i], node)
      )
    );

    const shiftParams = new ParameterLayer(
      scaleLayer.outputNodes.length,
      Vector.zero(scaleLayer.outputNodes.length)
    );

    const shiftLayer = new SimpleLayer(
      scaleLayer.outputNodes.map(
        (node, i) => new AddNode(shiftParams.outputNodes[i], node)
      )
    );

    super(shiftLayer.outputNodes);
  }
}
