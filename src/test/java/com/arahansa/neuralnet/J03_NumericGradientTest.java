package com.arahansa.neuralnet;

import mikera.matrixx.Matrix;
import org.junit.Before;
import org.junit.Test;

import java.util.function.Function;

/**
 * Created by jarvis on 2017. 4. 27..
 */
public class J03_NumericGradientTest {


    J03_NumericGradient numericGradient;

    @Before
    public void setup(){
        numericGradient = new J03_NumericGradient();
    }


    @Test
    public void 기울기테스트() throws Exception{
        // 편미분 공식
        Function<Matrix, Double> f = t -> Math.pow(t.get(0, 0), 2) + Math.pow(t.get(0, 1), 2);
        // 좌표
        Matrix matrix = new Matrix(1, 2);
        matrix.setElements(3, 4);

        // 계산
        Matrix numericGradient = this.numericGradient.getNumericGradient(f,matrix);
        System.out.println("(3,4) 에 대한 기울기 : \n"+numericGradient);

        matrix.setElements(1, 2);
        numericGradient = this.numericGradient.getNumericGradient(f,matrix);
        System.out.println("(1,2) 에 대한 기울기 \n"+numericGradient);

    }
}