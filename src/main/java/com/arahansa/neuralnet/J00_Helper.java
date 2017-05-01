package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.vectorz.AVector;
import org.springframework.stereotype.Component;

import java.util.Random;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Slf4j
@Component
public class J00_Helper {

    public static Matrix sigmoid(Matrix x){
        x.multiply(-1);
        x.exp();
        x.add(1);
        x.reciprocal();
        return x;
    }


    public static Matrix softmax(Matrix m){
        double max = m.elementMax();
        m.add(-max);
        m.exp();
        double sum = m.elementSum();
        m.divide(sum);
        return m;
    }

    /**
     * 두 행렬간의 곱을 구한다. Matrix lib 에서 못찾아서 내가 만든다. =ㅅ=;
     * @param x
     * @param y
     * @return
     */
    public static Matrix multipleTwoMatrix(Matrix x, Matrix y){
        if(x.getShape(1)!=y.getShape(0))
            throw new IllegalStateException("x column : "+x.getShape(1)+", y row :"+y.getShape(0)+" different.");
        Matrix multiplied = Matrix.create(x.getShape(0), y.getShape(1));
        for(int i=0;i<x.getShape(0);i++){
            for(int j=0;j<y.getShape(1);j++) {
                AVector row = x.getRowClone(i);
                AVector column = y.getColumnClone(j);
                row.multiply(column);

                double v = row.elementSum();
                multiplied.set(i,j,v);
            }
        }
        return multiplied;
    }

    /**
     * 정규분포를 따르는 행렬 얻기
     * http://mwultong.blogspot.com/2006/09/java-gaussian-gaussian-random-numbers.html
     * @param row
     * @param col
     * @return
     */
    public static Matrix getRadNMatrix(int row, int col){
        Random oRandom = new Random();
        Matrix m = Matrix.create(row, col);
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                m.set(i,j, oRandom.nextGaussian());
            }
        }
        return m;
    }

}
