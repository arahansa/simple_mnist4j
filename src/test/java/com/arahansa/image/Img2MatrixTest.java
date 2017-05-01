package com.arahansa.image;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by jarvis on 2017. 4. 27..
 */
public class Img2MatrixTest {

     @Test
     public void loadImgTest() throws Exception{

         Img2Matrix reader = new Img2Matrix();
         reader.getMatrix1Dimen("0.png");
         reader.getMatrix2Dimen("0.png");
         reader.getMatrix2Dimen("1.png");

     }


}