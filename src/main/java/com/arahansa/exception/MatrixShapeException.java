package com.arahansa.exception;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;

/**
 * Created by jarvis on 2017. 5. 5..
 */
@Slf4j
public class MatrixShapeException extends RuntimeException {

    public MatrixShapeException(){
        super();
    }
    public MatrixShapeException(Matrix x, Matrix y){
        super("x shape("+x.getShape(0)+","+x.getShape(1)
                +") , y shape("+y.getShape(0)+","+y.getShape(1)+") . can not calculate");
    }
}
