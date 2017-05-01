package com.arahansa.ml;

import mikera.matrixx.Matrix;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by jarvis on 2017. 4. 27..
 */
public class J05_SimpleNetTest {


    @Test
    public void 심플넷_기울기_구하기() throws Exception{
        J05_SimpleNet simpleNet = new J05_SimpleNet(null, null);


        Matrix w = Matrix.create(2,3);
        w.setElements(1,2,3,1,2,3);

        Matrix x = Matrix.create(1,2);
        x.setElements();

    }

}