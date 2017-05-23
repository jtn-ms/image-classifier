import caffe
import numpy as np
import yaml
from utils.timer import Timer
import config as cfg


class BatchTripletLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the BatchTripletLayer."""

        assert bottom[0].num == bottom[1].num, '{} != {}'.format(
            bottom[0].num, bottom[1].num)
        assert bottom[0].num == bottom[2].num, '{} != {}'.format(
            bottom[0].num, bottom[2].num)

        layer_params = yaml.load(self.param_str)
        self.alpha = layer_params['alpha']
	self.beta = layer_params['beta']
        self._timer = Timer()

        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # self._timer.tic()
        anchor = np.array(bottom[0].data)
        positive = np.array(bottom[1].data)
        negative = np.array(bottom[2].data)
        aps = np.sum((anchor - positive) ** 2, axis=1)
        ans = np.sum((anchor - negative) ** 2, axis=1)
	Mu_ap = np.sum(aps) / bottom[0].num
	Mu_an = np.sum(ans) / bottom[0].num
	sigma_ap = (aps - Mu_ap) ** 2
	sigma_an = (ans - Mu_an) ** 2
        #print 'ap' , aps
        #print 'an' , ans
	#print 'Mu_ap', Mu_ap
	#print 'sigma_ap',sigma_ap

        first_dist = self.alpha + aps - ans
        first_dist_hinge = np.maximum(first_dist, 0.0)
	second_dist_hinge = sigma_ap + sigma_an
	dist_hinge = (1.0 - self.beta) * first_dist_hinge + self.beta * second_dist_hinge
	#print 'alpha',self.alpha
	#print 'beta',self.beta
	#print 'first',first_dist_hinge
	#print 'second',second_dist_hinge
	#print 'hinge',dist_hinge
        # add semi-hard mining
        # if cfg.SEMI_HARD:
        #     semihard = np.asarray(np.less(aps, ans), dtype=np.float)
        #     dist_hinge *= semihard

        self.residual_list = np.asarray(first_dist_hinge > 0.0, dtype=np.float)
        loss = np.sum(dist_hinge) / bottom[0].num

        top[0].data[...] = loss
        # print 'loss' , loss
        # self._timer.toc()
        # print 'Loss:', self._timer.average_time

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            anchor = np.array(bottom[0].data)
            positive = np.array(bottom[1].data)
            negative = np.array(bottom[2].data)

            coeff = 2.0 * top[0].diff / bottom[0].num
            bottom_a = coeff * \
                np.dot(np.diag(self.residual_list), (negative - positive))
            bottom_p = coeff * \
                np.dot(np.diag(self.residual_list), (positive - anchor))
            bottom_n = coeff * \
                np.dot(np.diag(self.residual_list), (anchor - negative))

            bottom[0].diff[...] = bottom_a
            bottom[1].diff[...] = bottom_p
            bottom[2].diff[...] = bottom_n

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
