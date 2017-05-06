package com.arahansa.image;

import mikera.matrixx.Matrix;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by jarvis on 2017. 4. 27..
 */
public class Img2MatrixTest {

     @Test
     public void loadImgTest() throws Exception{

         Img2Matrix reader = new Img2Matrix();
         reader.getMatrix1Dimen("0.png");
         reader.getMatrix2Dimen("1.png");
         reader.getMatrix2Dimen("2.png");
         reader.getMatrix2Dimen("3.png");
         reader.getMatrix2Dimen("4.png");
         reader.getMatrix2Dimen("5.png");
         reader.getMatrix2Dimen("6.png");
         reader.getMatrix2Dimen("7.png");
         reader.getMatrix2Dimen("8.png");
         reader.getMatrix2Dimen("9.png");

         Map<Integer, Matrix> ttrainSampleMap = new LoadMnist().getTtrainSampleMap();
         System.out.println(ttrainSampleMap.get(0));

     }


}