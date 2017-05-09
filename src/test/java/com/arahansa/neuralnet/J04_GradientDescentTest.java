package com.arahansa.neuralnet;

import mikera.matrixx.Matrix;
import org.junit.Before;
import org.junit.Test;

import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.offset;


/**
 * Created by jarvis on 2017. 4. 27..
 */
public class J04_GradientDescentTest {


    J03_NumericGradient numericGradient;
    J04_GradientDescent gradientDescent;

    @Before
    public void setup(){
        numericGradient = new J03_NumericGradient();
        gradientDescent = new J04_GradientDescent(numericGradient);
    }

    @Test
    public void 경사하강법_테스트() throws Exception{
        // 편미분 공식
        Function<Matrix, Double> f = t -> Math.pow(t.get(0, 0), 2) + Math.pow(t.get(0, 1), 2);

        // 좌표
        Matrix m = new Matrix(1, 2);
        m.setElements(3, 4);

        Matrix gradientDescent = this.gradientDescent.getGradientDescent(f, m, 0.01, 100);
        System.out.println("final "+ gradientDescent);

        assertThat(gradientDescent.get(0,0)).isEqualTo(0.3978, offset(0.0001));
        assertThat(gradientDescent.get(0,1)).isEqualTo(0.5304, offset(0.0001));
    }




}