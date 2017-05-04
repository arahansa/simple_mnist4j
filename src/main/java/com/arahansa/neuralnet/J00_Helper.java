package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.vectorz.AVector;
import org.springframework.stereotype.Component;

import javax.swing.plaf.basic.BasicTableHeaderUI;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
        if(x.getShape(1)!=y.getShape(0)){
            log.info("x shape : ({}, {}), y shape : ({}, {})", x.getShape(0), x.getShape(1), y.getShape(0), y.getShape(1));
            throw new IllegalStateException("x column : "+x.getShape(1)+", y row :"+y.getShape(0)+" different.");
        }

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


    /**
     * x, t 를 받아서 batch 사이즈만큼 랜덤사이즈를 가진 x,t 로 돌려준다.
     * (60000, 784) , (60000, 10) -> (100, 784)(100, 10) 이렇게도..
     * @param x
     * @param t
     * @param batch_size
     * @return
     */
    public static Map<String, Matrix> getBatchMatrix(Matrix x, Matrix t, int batch_size){
        if(x.getShape(0)!=t.getShape(0))
            throw new IllegalStateException("x , t size err");

        Map<String, Matrix> map = new HashMap<>();
        Matrix x_batch = Matrix.create(batch_size, x.getShape(1));
        Matrix t_batch = Matrix.create(batch_size, t.getShape(1));


        generateRandomUniqueNumbers(x.getShape(0), batch_size).forEach(n->{
            // log.info("n : {}", n);
            x_batch.add(x.getRow(n));
            t_batch.add(t.getRow(n));
        });

        map.put("x_batch", x_batch);
        map.put("t_batch", t_batch);
        return map;
    }

    /**
     * 랜덤 유니크 숫자 돌려주기
     * @param max
     * @param batch_size
     * @return
     */
    public static List<Integer> generateRandomUniqueNumbers(int max, int batch_size){
        List<Integer> range = IntStream.range(0, max).boxed()
                .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(range);
        return range.subList(0, batch_size);
    }


}
