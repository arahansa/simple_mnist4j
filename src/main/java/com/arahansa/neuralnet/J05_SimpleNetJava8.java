package com.arahansa.neuralnet;

import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.function.BiFunction;

/**
 * Created by jarvis on 2017. 5. 2..
 */
@Slf4j
public class J05_SimpleNetJava8{

    @Setter
    Matrix w;

    public Matrix predict(Matrix x){
        return J00_Helper.multipleTwoMatrix(x, this.w);
    }

    BiFunction<Matrix, Matrix, Double> lossFunc = (x,t)->{
        Matrix z = predict(x);
        Matrix y = J00_Helper.softmax(z);
        return J01_CostFunction.getCrossEntropyErr(y, t);
    };

}
