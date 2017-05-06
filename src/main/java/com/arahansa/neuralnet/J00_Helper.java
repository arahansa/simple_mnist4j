package com.arahansa.neuralnet;

import com.arahansa.exception.MatrixShapeException;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.vectorz.AVector;
import mikera.vectorz.impl.AStridedVector;
import mikera.vectorz.impl.ArraySubVector;
import org.springframework.stereotype.Component;
import sun.tools.jconsole.MaximizableInternalFrame;

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

    /*
    def sigmoid_grad(x):
            return (1.0 - sigmoid(x)) * sigmoid(x)
    */

    public static Matrix sigmoid_grad(Matrix x){
        Matrix x_copy = x.copy();

        Matrix sigmoidX = sigmoid(x_copy);
        sigmoidX.multiply(-1);

        Matrix m1 = Matrix.create(x.getShape(0), x.getShape(1));
        m1.add(1);
        m1.add(sigmoidX); // 1 - sigmoid(x)


        m1.multiply(sigmoid(x.copy())); // * sigmoid(x)
        return m1;
    }


    public static Matrix softmax(Matrix m){
        Matrix x = transpose(m);

        Matrix max = max0Dimen(x);
        max.multiply(-1);
        x = add(x, max); // x = x- np.max(x)

        x.exp();

        Matrix xExp = x.copy();
        Matrix npSum = sum(xExp);
        x = divide(x, npSum);
        return transpose(x);
    }

    /**
     * np.transpose
     * @param x
     * @return
     */
    public static Matrix transpose(Matrix x){
        if(x.getShape(0)==x.getShape(1)){
            Matrix copy = x.copy();
            copy.transposeInPlace();
            return copy;
        }
        Matrix r = Matrix.create(x.getShape(1), x.getShape(0));
        for(int i=0;i<x.getShape(1);i++)
            r.setRow(i, x.getColumn(i));
        return r;
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
     * matrix shape 이 안 맞을때 divide
     * @param x
     * @param y
     * @return
     */
    public static Matrix divideTwoMatrix(Matrix x, Matrix y){
        if(x.getShape(1)!=y.getShape(1)){
            log.info("x shape : ({}, {}), y shape : ({}, {})", x.getShape(0), x.getShape(1), y.getShape(0), y.getShape(1));
            throw new IllegalStateException("x column : "+x.getShape(1)+", y row :"+y.getShape(0)+" different.");
        }

        Matrix divided = x.copy();
        ArraySubVector yRow = y.getRow(0);
        for(int i=0;i<x.getShape(0);i++)
            divided.getRow(i).divide(yRow);
        return divided;
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


    // np.sum 차원 축소
    public static Matrix sum(Matrix x){
        int size = x.getShape(1);
        Matrix z = transpose(x);

        Matrix y = Matrix.create(1, size);

        for(int i=0;i<size;i++){
            double v = z.getRow(i).elementSum();
            y.set(0, i, v);
        }
        return y;
    }

    /**
     * [5,4,1,
     *  2,3,4,
     *  4,5,2]
     *
     *  => [5,5,4] (세로에서 뽑는다)
     * @param x
     * @return
     */
    public static Matrix max0Dimen(Matrix x){
        int size = x.getShape(1);
        Matrix result = Matrix.create(1, size);

        for(int i=0;i < size; i++){
            double v = x.getColumn(i).elementMax();
            result.set(0,i,v);
        }
        return result;
    }

    /**
     * [5,4,1,
     *  2,3,4,
     *  4,5,2]
     *
     *  => [5,4,5] (가로에서 뽑는다)
     * @param x
     * @return
     */
    public static Matrix max1Dimen(Matrix x){
        int size = x.getShape(0);
        Matrix result = Matrix.create(1, size);

        for(int i=0;i < size; i++){
            double v = x.getRow(i).elementMax();
            result.set(0,i,v);
        }
        return result;
    }

    /**
     *
     * @param y
     * @param t
     * @return
     */
    public static Matrix collectByTmaxElem(Matrix y, Matrix t){
        if(!y.isSameShape(t)) throw new MatrixShapeException(y,t);
        int size = y.getShape(0);
        Matrix result = Matrix.create(1, size);
        for(int i=0;i<size;i++){
            int j = t.getRow(i).maxElementIndex();
            result.set(0, i, y.get(i,j));
        }
        return result;
    }

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public static Matrix add(Matrix x, Matrix y){
        Matrix copy = x.copy();
        if(y.getShape(0)==1 && y.getShape(1)==1){
            copy.add(y.get(0,0));
            return copy;
        }
        if(x.getShape(0)==1 && x.getShape(1)==1){
            copy = y.copy();
            copy.add(x.get(0,0));
            return copy;
        }
        // x [10, 100] y : [1, 100]
        if(y.getShape(0)==1) {
            if (x.getShape(1) != y.getShape(1)) throw new MatrixShapeException(x, y);
            ArraySubVector yRow = y.getRow(0);
            for (int i = 0; i < copy.getShape(0); i++) {
                copy.getRow(i).add(yRow);
            }
            return copy;
        }
        // x[ 100, 10] y [ 100, 1]
        if(y.getShape(1)==1){
            if(x.getShape(0)!=y.getShape(0)) throw new MatrixShapeException(x, y);
            AStridedVector yColumn = y.getColumn(0);
            for(int i=0;i<copy.getShape(1);i++){
                copy.getColumn(i).add(yColumn);
            }
            return copy;
        }
        copy.add(y);
        return copy;
    }

    public static Matrix divide(Matrix x, Matrix y){
        Matrix copy = x.copy();
        if(y.getShape(0)==1 && y.getShape(1)==1){
            copy.divide(y.get(0,0));
            return copy;
        }
        if(x.getShape(0)==1 && x.getShape(1)==1){
            copy = y.copy();
            copy.divide(x.get(0,0));
            return copy;
        }

        // x [10, 100] y : [1, 100]
        if(y.getShape(0)==1) {
            if (x.getShape(1) != y.getShape(1)) throw new MatrixShapeException(x, y);
            ArraySubVector yRow = y.getRow(0);
            for (int i = 0; i < copy.getShape(0); i++) {
                copy.getRow(i).divide(yRow);
            }
            return copy;
        }
        // x[ 100, 10] y [ 100, 1]
        if(y.getShape(1)==1){
            if(x.getShape(0)!=y.getShape(0)) throw new MatrixShapeException(x, y);
            AStridedVector yColumn = y.getColumn(0);
            for(int i=0;i<copy.getShape(1);i++){
                copy.getColumn(i).divide(yColumn);
            }
            return copy;
        }
        copy.divide(y);
        return copy;
    }

    /**
     * np.argmax.
     * axis 는 현재 0 과 1일 상황만을 가정한다
     * @param x
     * @param axis
     * @return
     */
    public static Matrix argmax(Matrix x, int axis){
        if(axis !=0 && axis != 1)
            throw new IllegalStateException();

        int size = axis==0 ? x.getShape(1) : x.getShape(0);
        Matrix m = Matrix.create(1, size);

        if(axis==0){
            for(int i=0;i<size;i++){
                int maxIdx = x.getColumn(i).maxElementIndex();
                m.set(0,i, maxIdx);
            }
        }else{
            for(int i=0;i<size;i++){
                int maxIdx = x.getRow(i).maxElementIndex();
                m.set(0,i, maxIdx);
            }
        }
        return m;
    }


}
