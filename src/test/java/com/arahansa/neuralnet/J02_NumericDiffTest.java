package com.arahansa.neuralnet;

import org.junit.Before;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.AssertionsForClassTypes.offset;


/**
 * Created by jarvis on 2017. 4. 27..
 */
public class J02_NumericDiffTest {

    J02_NumericDiff numericDiff;

    @Before
    public void setup(){
        numericDiff = new J02_NumericDiff();
    }

    @Test
    public void 몸무게증가() throws Exception{
        double numericDiff = this.numericDiff.getNumericDiff(x -> 1*x, 1);
        System.out.println("1x 미분 값 : "+numericDiff);

        numericDiff = this.numericDiff.getNumericDiff(x -> 2*x, 1);
        System.out.println("2x 미분 값 : "+numericDiff);

    }

    @Test
    public void 수치미분의_예() throws Exception{
        double numericDiff = this.numericDiff.getNumericDiff(d -> 0.01 * Math.pow(d, 2) + 0.1 * d, 5);
        System.out.println("0.01x(제곱) + 0.1x 미분 값 : "+numericDiff);
    }


    @Test
    public void 편미분_테스트() throws Exception{
        double numericDiff = this.numericDiff.getNumericDiff(x -> Math.pow(x, 2) + Math.pow(4, 2), 3.0);
        System.out.println("(3,4) 에서 3에 대한 편미분 (x0제곱 + 4제곱) : "+numericDiff);
        assertThat(numericDiff).isEqualTo(6.00, offset(0.01));

        numericDiff = this.numericDiff.getNumericDiff(x -> Math.pow(3, 2) + Math.pow(x, 2) , 4.0);
        System.out.println("(3,4) 에서 4에 대한 편미분 (3제곱 + x1제곱) : "+numericDiff);
        assertThat(numericDiff).isEqualTo(7.99, offset(0.01));
    }



}