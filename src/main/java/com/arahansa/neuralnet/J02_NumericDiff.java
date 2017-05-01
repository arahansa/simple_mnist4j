package com.arahansa.neuralnet;

import org.springframework.stereotype.Component;

import java.util.function.Function;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Component
public class J02_NumericDiff {

    /**
     * 수치 미분
     * @param func
     * @param x
     * @return
     */
    public double getNumericDiff(Function<Double, Double> func, double x){
        double h = 1e-4;
        return (func.apply(x+h) - func.apply(x-h)) / (2 * h);
    }
}
