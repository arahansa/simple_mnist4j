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
    public void 하나의값에대한_편미분테스트() throws Exception{
        double numericDiff = this.numericDiff.getNumericDiff(d -> Math.pow(d, 2) + Math.pow(4, 2), 3.0);
        assertThat(numericDiff).isEqualTo(6.00, offset(0.2));

        numericDiff = this.numericDiff.getNumericDiff(d -> Math.pow(d, 2) + Math.pow(3, 2), 4.0);
        assertThat(numericDiff).isEqualTo(7.99, offset(0.2));
    }

    static class MyClass{

    }


}