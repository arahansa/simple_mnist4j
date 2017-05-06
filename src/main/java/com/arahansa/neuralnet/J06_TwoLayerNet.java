package com.arahansa.neuralnet;

import com.arahansa.data.Grad;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

import static com.arahansa.neuralnet.J00_Helper.*;

/**
 * Created by arahansa on 2017-05-02.
 */
@Slf4j
@Component
public class J06_TwoLayerNet {

    Matrix W1, W2, b1, b2;

    /**
     * W1 ( input_size, hidden_size)
     *
     * @param input_size
     * @param hidden_size
     * @param output_size
     * @param weight_init_std
     */
    public J06_TwoLayerNet(int input_size, int hidden_size, int output_size, Float weight_init_std){
        if(weight_init_std==null) weight_init_std = 0.01f;

        W1 = J00_Helper.getRadNMatrix(input_size, hidden_size);
        W1.multiply(weight_init_std);
        b1= Matrix.create(1, hidden_size);

        W2 = J00_Helper.getRadNMatrix(hidden_size, output_size);
        W2.multiply(weight_init_std);
        b2= Matrix.create(1, output_size);
    }

    public void setW1(Matrix W1){
        this.W1 = W1.copy();
    }
    public void setW2(Matrix W2){
        this.W2 = W2.copy();
    }
    public Matrix getW1(){return this.W1;}
    public Matrix getW2(){return this.W2;}

    public Matrix predict(Matrix x){
        Matrix a1 = multipleTwoMatrix(x, W1);
        a1.add(b1.getRow(0));

        Matrix z1 = sigmoid(a1);
        Matrix a2 = multipleTwoMatrix(z1, W2);
        a2.add(b2.getRow(0));

        return softmax(a2);
    }

    BiFunction<Matrix, Matrix, Double> lossFunc = (x, t)->{
        Matrix y = predict(x);
        double crossEntropyErr = J01_CostFunction.getCrossEntropyErr4Batch(y, t);
        return crossEntropyErr;
    };

    public double accuracy(Matrix x, Matrix t){
        Matrix y = predict(x);
        if(!t.isSameShape(y)){
            log.warn("t shape ({}, {}) , y shape ( {}, {} ) ", t.getShape(0), t.getShape(1), y.getShape(0), y.getShape(1));
            throw new IllegalStateException("different y , t size ");
        }
        return calcul_accracy(y, t);
    }

    double calcul_accracy(Matrix y, Matrix t){
        int sameCount = 0;
        int size = y.getShape(0);
        for(int i = 0; i < size; i++){
            if(y.getRow(i).maxElementIndex()==t.getRow(i).maxElementIndex()){
                sameCount++;
            }
        }
        double accuracy = sameCount / new Double(size);
        return accuracy;
    }

    public Grad numerical_gradient(Matrix x, Matrix t){
        Grad grad = new Grad();
        Matrix w1 = J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.W1);
        grad.setW1(w1);
        grad.setB1(J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.b1));

        Context.local.set("b1");
        grad.setW2(J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.W2));
        Context.local.set(null);

        grad.setB2(J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.b2));

        return grad;
    }


    /*
    def gradient(self, x, t):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
            y = softmax(a2)

    # backward
    dy = (y - t) / batch_num
    print("z1 : ", z1)
    print("z1 t : ", z1.T)
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)

    da1 = np.dot(dy, W2.T)
    dz1 = sigmoid_grad(a1) * da1
    grads['W1'] = np.dot(x.T, dz1)
    grads['b1'] = np.sum(dz1, axis=0)

    return grads
    */


    /**
     * 역전파 계산
     * @param x
     * @param t
     * @return
     */
    public Grad gradient(Matrix x, Matrix t){
        Matrix x_copy = x.copy();

        Grad grad = new Grad();
        int batch_num = x.getShape(0);

        // forward
        Matrix a1 = multipleTwoMatrix(x, this.W1);
        log.info("a1 : {}", a1);
        Matrix a1_copy = a1.copy();
        a1.add(this.b1.getRow(0));
        Matrix z1 = sigmoid(a1);
        Matrix a2 = multipleTwoMatrix(z1, this.W2);
        a2.add(this.b2.getRow(0));
        log.info("a2 : {}", a2);
        Matrix y = softmax(a2);
        log.info("softmax y : {}", y);

        Matrix t_back = t.copy();
        t_back.multiply(-1);
        y.add(t_back);
        y.divide(batch_num); // dy

        z1.transposeInPlace();
        // backward
        grad.setW2(multipleTwoMatrix(z1, y));
        grad.setB2(sum(y));

        Matrix W2_copy = W2.clone();
        W2_copy.transposeInPlace();
        Matrix da1 = multipleTwoMatrix(y, W2_copy);
        Matrix dz1 = sigmoid_grad(a1_copy);
        dz1.multiply(da1);

        x_copy.transposeInPlace();
        grad.setW1(multipleTwoMatrix(x_copy, dz1));
        grad.setB1(sum(dz1));
        return grad;
    }



    public void renewParams(Grad grad, Double learning_rate){
        log.info("W1 shape : {}, gradW1 shape: {} ",this.W1.getShape(), grad.getW1().getShape());
        log.info("before renew param 1 : {}", this.W1.getRow(0));
        Matrix w1 = grad.getW1();
        w1.multiply(-1*learning_rate);
        log.info("grad w1 :{}", w1.getRow(0));
        W1.add(w1);
        log.info("after renew param 1 : {}", this.W1.getRow(0));
        Matrix w2 = grad.getW2();
        w2.multiply(-1*learning_rate);
        W2.add(w2);

        Matrix b1 = grad.getB1();
        b1.multiply(-1*learning_rate);
        this.b1.add(b1);

        Matrix b2 = grad.getB2();
        b2.multiply(-1*learning_rate);
        this.b2.add(b2);

    }

    public static class Context {
        public static ThreadLocal<String> local = new ThreadLocal<>();
    }





}
