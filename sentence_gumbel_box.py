import torch
import numpy as np
import torch.nn as nn
from basic_box import Box
from utils import ExpEi
import torch.nn.functional as F
from torch.distributions import uniform

from transformers import BertModel

euler_gamma = 0.57721566490153286060

class SentenceBertGumbelBox(nn.Module):
	def __init__(
			self,
			bert_name_or_path,
			box_dim,
			softplus_temp,
			gumbel_beta,
			scale,
			device
		):
		super(SentenceBertGumbelBox, self).__init__()
		self.bert = BertModel.from_pretrained(bert_name_or_path)
		self.center = nn.Linear(self.bert.config.hidden_size, box_dim)
		self.delta  = nn.Linear(self.bert.config.hidden_size, box_dim)

		self.temperature = softplus_temp
		self.gumbel_beta = gumbel_beta
		self.scale = scale

		self.device = device
		self.to(self.device)

	def forward(self, s1_ids, s1_msk, s2_ids, s2_msk):
		"""Returns box embeddings for ids"""
		s1_embed = self.pooling(self.bert(s1_ids, s1_msk), s1_msk)
		s1_box = Box(center_embed = self.center(s1_embed), delta_embed = self.delta(s1_embed))

		s2_embed = self.pooling(self.bert(s2_ids, s2_msk), s2_msk)
		s2_box = Box(center_embed = self.center(s2_embed), delta_embed = self.delta(s2_embed))

		pos_predictions = self.get_cond_probs(s1_box, s2_box)
		neg_prediction = torch.ones(pos_predictions.size()).to(self.device) - pos_predictions
		prediction = torch.stack([neg_prediction, pos_predictions], dim=1)
		return prediction
	
	def pooling(self, bert_out):
		# mean pooling
		return torch.mean(bert_out[0], dim=1)

	def volumes(self, boxes: Box):
		eps = torch.finfo(boxes.min_embed.dtype).tiny  # type: ignore

		if isinstance(self.scale, float):
			s = torch.tensor(self.scale)
		else:
			s = self.scale

		return torch.sum(
			torch.log(F.softplus(boxes.max_embed - boxes.min_embed - 2*euler_gamma*self.gumbel_beta, beta=self.temperature).clamp_min(eps)),
			dim=-1) + torch.log(s)

	def intersection(self, boxes1, boxes2):
		z = self.gumbel_beta * torch.logsumexp(torch.stack((boxes1.min_embed / self.gumbel_beta, boxes2.min_embed / self.gumbel_beta)), 0)
		z = torch.max(z, torch.max(boxes1.min_embed, boxes2.min_embed)) # This line is for numerical stability (you could skip it if you are not facing any issues)

		Z = - self.gumbel_beta * torch.logsumexp(torch.stack((-boxes1.max_embed / self.gumbel_beta, -boxes2.max_embed / self.gumbel_beta)), 0)
		Z = torch.min(Z, torch.min(boxes1.max_embed, boxes2.max_embed)) # This line is for numerical stability (you could skip it if you are not facing any issues)

		intersection_box = Box(z, Z)
		return intersection_box

	def get_cond_probs(self, boxes1, boxes2):
		# log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
		# log_box2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
		log_intersection = torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), np.log(1e-10), np.log(1e4))
		log_box2 = torch.clamp(self.volumes(boxes2), np.log(1e-10), np.log(1e4))
		return torch.exp(log_intersection - log_box2)
