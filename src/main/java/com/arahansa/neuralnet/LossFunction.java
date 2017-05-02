package com.arahansa.neuralnet;

import mikera.matrixx.Matrix;

/**
 * Created by jarvis on 2017. 5. 2..
 */
@FunctionalInterface
public interface LossFunction {
    double loss(Matrix x, Matrix t);
}
