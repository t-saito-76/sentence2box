import torch

class Box:
	def __init__(self, min_embed, max_embed, center_embed, delta_embed):
		self.min_embed: torch.Tensor
		self.max_embed: torch.Tensor
		self.center_embed: torch.Tensor
		self.delta_embed: torch.Tensor

		if       all([min_embed, max_embed]) and not any([center_embed, delta_embed]):
			self.min_embed = min_embed  
			self.max_embed = max_embed
			self.delta_embed  = (max_embed - min_embed) / 2
			self.center_embed = (max_embed + min_embed) / 2
		elif not any([min_embed, max_embed]) and     all([center_embed, delta_embed]):
			self.min_embed = center_embed - delta_embed  
			self.max_embed = center_embed + delta_embed
			self.delta_embed  = delta_embed
			self.center_embed = center_embed
		else:
			ValueError("Box must initialize with [min_embed, max_embed] or [center_embed, delta_embed]")
