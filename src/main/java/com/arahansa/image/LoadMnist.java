package com.arahansa.image;

import mikera.matrixx.Matrix;
import mikera.vectorz.impl.ArraySubVector;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by arahansa on 2017-05-03.
 */
    public class LoadMnist {

    // x_train, t_train, x_test, t_test
    // (60000, 784) ( 60000, 10), (10000, 784) ,  (10000, 10)
    // 여기서는 (10000, 15) , (10000, 10), (3000, 15, 3000, 10) 으로 만들어보자.
    public Map<String, Matrix> loadMnist(){
        Map<Integer, Matrix> xtrainSampleMap = getXtrainSampleMap();
        Map<Integer, Matrix> ttrainSampleMap = getTtrainSampleMap();

        Map<String, Matrix> map = new HashMap<>();

        Matrix x_train = Matrix.create(10000, 15);
        Matrix t_train = Matrix.create(10000, 10);

        Matrix x_test = Matrix.create(3000, 15);
        Matrix t_test = Matrix.create(3000, 10);


        for(int i=0;i<10000;i++){
            ArraySubVector xRow = xtrainSampleMap.get(i % 10).getRow(0);
            ArraySubVector tRow = ttrainSampleMap.get(i % 10).getRow(0);
            x_train.setRow(i, xRow);
            t_train.setRow(i, tRow);
            if(i<3000){
                x_test.setRow(i, xRow);
                t_test.setRow(i, tRow);
            }
        }

        map.put("x_train", x_train);
        map.put("t_train", t_train);
        map.put("x_test", x_test);
        map.put("t_test", t_test);
        return map;
    }

    /**
     * x_train 맵 가져오는 정도로 가져옴..
     * @return
     */
    public Map<Integer, Matrix> getXtrainSampleMap(){
        Img2Matrix img2Matrix = new Img2Matrix();

        Map<Integer, Matrix> m = new HashMap<>();
        for(int i=0;i<=9;i++){
            try {
                Matrix matrix = img2Matrix.getMatrix1Dimen(i + ".png");
                m.put(i, matrix);
            } catch (IOException e) {
                e.printStackTrace();
                throw new IllegalStateException();
            }
        }
        return m;
    }

    /**
     * t_train 맵 정답 용도로 가져옴
     * @return
     */
    public Map<Integer, Matrix> getTtrainSampleMap(){
        Map<Integer, Matrix> m = new HashMap<>();
        for(int i=0;i<=9;i++){
            Matrix matrix = Matrix.create(1, 10);
            matrix.set(0, i, 1);
            m.put(i, matrix);
        }
        return m;
    }



}
