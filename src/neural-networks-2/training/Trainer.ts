import { DataSet } from "../../data/DataSet";
import { Matrix } from "../../matrix/Matrix";
import { NeuralNetwork } from "../networks/NeuralNetwork";
import { TrainingMonitor } from "./TrainingMonitor";

export abstract class Trainer {
  public abstract train<T extends object>(
    network: NeuralNetwork,
    dataSet: DataSet<T>,
    predictors: (entry: T) => Matrix,
    responses: (entry: T) => Matrix,
    monitor?: TrainingMonitor<T>
  ): void;
}
