import math
import torch
from torch import nn

class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.

    - This module splits each image into small patches (like puzzle pieces).
    - Each patch is the transformed (projected) into a vector which is a 1D representation of size "hidden_size".
    - This entire operation is done through a single convolution laeyer that moves with a stride equal to the patch size.
    - Ultimately, we end up with a batch of patch embeddings for each image:

    Shape: (batch_size, num_patches, hidden_size)
    """

    def __init__(self, config):
        super().__init__()
        
        # Extract hyperparameters from config
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        
        # Calculate the number of patches from the image size and patch size
        # Total patches = (horizontal_patches) * (vertical_patches)
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Create a projection layer which is a Conv2D that effectively does
        #   1) Extracts each patch of size (patch_size x patch_size),
        #   2) Learns a linear mapping to 'hidden_size' channels (so each patch becomes a vector of length hidden_size).
        # The kernel_size and stride both equal to patch_size ensures non-overlapping patches.
        # In the paper, the image_size = 224, patch_size = 16, hidden_size = 768, num_channels = 3
        self.projection = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_channels, image_height, image_width)
        
        Returns:
            A 3D tensor of shape (batch_size, num_patches, hidden_size)
            where each row is an embedded patch of length 'hidden_size'.
        """

        # 1) Apply the convolution to produce patch embeddings
        # Output shape: (batch_size, hidden_size, image_size/patch_size, image_size/patch_size)
        # Example: input shape (batch_size, 3, 224, 224) after projection, 
        #        shape becomes (batch_size, 768, 14, 14)
        x = self.projection(x)
        
        # 2) Flatten the (height, width) dimension into one dimension.
        #   shape -> (batch_size, hidden_size, num_patches) 
        #   where num_patches = (image_size / patch_size) * (image_size / patch_size)
        # Example: shape (batch_size, 768, 14*14) = (batch_size, 768, 196).
        x = x.flatten(2)    # merges dimension 2 and 3 => shape: (batch_size, hidden_size, num_patches)
        
        # 3) Transpose to get (batch_size, num_patches, hidden_size),
        #   so the "sequence" dimension (num_patches) is in the middle, matching Transformer convention.
        # Example: shape (batch_size, 196, 768)
        # That means we have 196 patches per image, each is a 768-dim vector.
        x = x.transpose(1, 2)   # shape: (batch_size, num_patches, hidden_size)
        
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.

    Steps:
    - First, we use PatchEmbeddings to get a sequence of patch vectors.
    - We have a special [CLS] token that we prepend to the patch embeddings.
        - the [CLS] token is added to the beginning of the input sequence (like BERT) 
        and is used to classify the entire sequence.
    - Then we add learnable position embeddings to each token.
    - Finally, we optionally apply dropout, as a form of regularization.

    Shape elolution:
    1 - from Input images: (batch_size, num_channnels, image_height, image_width), e.g., (B, 3, 224, 224)
    2 - PatchEmbeddings Output: (batch_size, num_patches, hidden_size), e.g., (B, 196, 768) in case patch size is 14x14.
    3 - [CLS] token expansion: (batch_size, 1, hidden_size)
    4 - Concatenation: (batch_size, num_patches + 1, hidden_size), e.g., (B, 197, 768)
    5 - Add position embeddings: shape remains (batch_size, num_patches + 1, hidden_size)
    6 - Dropout: shape still (batch_size, num_patch + 1, hidden_size)
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1) Patch embeddings:
        #   turns an input image of shape (batch_size, num_channels, image_height, image_width)
        #   into a sequence of patch vectors of shape (batch_size, num_patches, hidden_size)
        self.patch_embeddings = PatchEmbeddings(config)
        
        # 2) Create a learnable [CLS] token. Dimensions: (1, 1, hidden_size)
        #   The "1, 1" shape is so it can be expanded to any batch_size.
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config["hidden_size"])
        )
        
        # 3) Position embeddings:
        #   We need a separate embedding for each patch plus the [CLS] token, so total tokens = num_patches + 1
        #   Dimension of position_embeddings: (1, num_patches + 1, hidden_size)
        #   - The "1" in the first dimension again allows easy broadcast across batch_size
        self.position_embeddings = nn.Parameter(
                torch.randn(
                    1,
                    self.patch_embeddings.num_patches + 1,
                    config["hidden_size"]
                )
            )
        
        # 4) Optional dropout for regularization which help reduce overfitting
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])


    def forward(self, x):
        """
        Args:
            x : Input images of shape (batch_size, num_channels, image_height, image_width)

        Returns:
            A 3D tensor of shape (batch_size, num_patches + 1, hidden_size)
            with the [CLS] token included and position embeddings added.
        """

        # 1) Get patch embeddings from the PatchEmbeddings module.
        x = self.patch_embeddings(x)    # shape (batch_size, num_patches, hidden_size)

        # Extract the batch_size from the resulting shape after patch_embeddings
        batch_size, _, _ = x.size()
        
        # 2) Expand the learnable [CLS] token to the whole batch:
        #   shape => (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 3) Prepend the [CLS] token to the beginning of the the patch embeddings
        #   Now x => shape (batch_size, (num_patches + 1), hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4) Add position embeddings
        #   position_embeddings has shape (1, (num_patches + 1), hidden_size),
        #   which is broadcasted to (batch_size, (num_patches + 1), hidden_size)
        x = x + self.position_embeddings
        
        # 5) Dropout (shape remains the same)
        x = self.dropout(x)
        
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.

    This module is used in the MultiHeadAttention module.

    - AttentionHead does one "head" of the self-attention operation.
    - Recall that Multi-Head Attention splits the hidden dimension into multiple heads.
    Each head learns separate projections (Q, K, V) and handles a portion of the 'hidden_size'.
    - The final output of each head is then concatenated in 'MultiHeadAttention'.

    Shape evolution:
    1 - Input: (batch_size, seq_length, hidden_size)
    2 - Query/Key/Value: each (batch_size, seq_length, attention_head_size)
    3 - Attention Scores: computed by `query @ key^T` shape (batch_size, seq_length, seq_length)
    4 - Attention Probs: results after softmax along the last dimension, shape (batch_size, seq_length, seq_length)
    5 - Attention Output: `(attention_probs @ value)`), shape => (batch_size, seq_length, attention_head_size)


    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # Create the query, key, and value projection layers. These layers do"
        #   query = x + Wq (transform hidden_size -> attention_head_size)
        #   key   = x + Wk (transform hidden_size -> attention_head_size)
        #   query = x + Wv (transform hidden_size -> attention_head_size)
        # If 'bias=True', each linear layer has an addictive bias term too.
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key   = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        # Dropout for attention probabilities (helps reduce overfitting)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, hidden_size)

        Returns:
            (attention_output, attention_probs)

            attention_output:
                shape => (batch_size, sequence_length, attention_head_size)

            attention_probs:
                shape => (batch_size, sequence_length, sequence_length)
                these are the actual attention weights learned for each token pair.
        """
        
        # Project the input into query, key, and value for this single head,
        # which uses the same input to generate the query, key, and value.
        # Shapes after each projection => (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key   = self.key(x)
        value = self.value(x)
        
        # Calculate the attention scores:
        #   attention_scores = Q * K^T / sqrt(attention_head_size)
        #   => shape: (batch_size, sequence_length, sequence_length)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Convert these scores to probabilities using softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply dropout on the attention probabilities
        attention_probs = self.dropout(attention_probs)

        # Weighted sum: attention_probs * V
        # => shape: (batch_size, sequence_length, attention_head_size)
        attention_output = torch.matmul(attention_probs, value)

        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module (the standard approach, similar to the one in the paper)
    This module is used in the TransformerEncoder module.

    This module orchestrates multiple individual `AttentionHead` in parallel,
    then merges (concatenates) their outputs.

    Explaintion:
    - Instead of a single 'AttentionHead', we use multiple heads in parallel.
    - Each head attends to different representation subspaces.
    - Then we concatenate the outputs of all heads and project them back to the original dimensionality (hidden_size):

    Shape evolution:
    1 - Input (batch_size, sequence_length, hidden_size)
    2 - Heads:
        - Each head sees the entire (batch_size, sequence_length, hidden_size) as input, but within each head's linear layers,
            only `attention_head_size` is used for the Q, K, V dimension.
        - The single-head output shape is `(batch_size, sequence_length, attention_head_size)`.

    3 - Concatenate along the last dimension => (batch_size, sequence_length, all_head_size)` which is `(batch_size, sequence_length, hidden_size) in total.
    4 - Lineaer output projection => shape remains (batch_size, sequence_length, hidden_size).

    5 - (Optional) If `output_attentions=True`, we also gather attention probability matrices from each head,
            stacking them into shape (batch_size, num_attention_heads, seq_len, seq_len)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        
        # The attention_head_size is how many hidden features that each head handles.
        # Typically, attention_head_size = hidden_size // num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        # Multiplying them together just get us back to hidden_size, e.g., 8 heads * 96 dims/head = 768
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]

        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        
        # After each head produces an output, we concatenate them,
        # then project back to hidden_size dimension
        # (In standard Transformers, this is often called W^0, the output projection)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

        # Dropuut for the final projected output
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        """
        Args:
            x: shape => (batch_size, sequence_length, hidden_size)
            output_attentions: if True, return attention matrices for analysis

        Returns:
            (attention_output, attention_probs or None)
                - attention_output => (batch_size, sequence_length, hidden_size)
                - attention_probs => either None or a stacked version (batch_size, num_heads, seq_len, seq_len)

        """
        # Calculate the attention output for each attention head
        # EAch 'head' returns (head_output, head_probs)
        attention_outputs = [head(x) for head in self.heads]
        # 'attention_outputs' is a list of length num_attention_heads
        # Each 'head' returns (head_output, head_probs)
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # shape => (batch_size, sequence_length, all_head_size) i.e. (batch_size, seq_len, hidden_size)

        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)

        # Apply dropout
        attention_output = self.output_dropout(attention_output)
        
        # Return the attention output and the attention probabilities (if needed)
        if not output_attentions:
            return (attention_output, None)
        else:
            # Stack the attention_probs across heads to shape => (batch_size, num_heads, seq_len, seq_len)
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    ---

    This is a varient of multi-head attention that combines the Q/K/V projections for **all heads**
    into a single linear layer (plus the usual output projection).

    Combining Q/K/V into a single linear call is faster, using fewer kernel launches.
    It's also more direct and less boilerplate from creating each head's linear layers separately.
    Sometimes it's slightly more memory-efficient, as we only hold one big weight matrix and do
    one big matmul.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 1) Single projection: (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, 3 * all_head_size)
        qkv = self.qkv_projection(x)
        
        # 2) Split the projected query, key, and value into query, key, and value
        # each => (batch_size, seq_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        
        # 3) Reshape the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        #       so we can do multi-head attention in parallel
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key     = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Compute the attention scores, apply dropout, multiply by value
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Compute the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # shape => (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # -> (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                        .contiguous() \
                                        .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). 
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    ---
    This is a variant of the standard GELU activation function. The formula is an approximation (tanh-based) instead of the 
    more numerically-heavy error func. So it can be faster than the extract error function.

    ---
    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * 
            (input + 0.044715 * torch.pow(input, 3.0))
        ))



class MLP(nn.Module):
    """
    A multi-layer perceptron module.

    This module also referred to as the 'feed-forward network' in Transformer blocks.
    It transforms each token's hidden vector independently (applies the same MLP to each token).
    Linear -> Activation -> Linear -> Dropout.

    The original paper used a GELU activation function but here we use a variant of it.

    Shape evolution:
    1 - Input: (batch_size, seq_len, hidden_size)
    2 - After `dens_1`: (batch_size, seq_len, intermediate_size)
    3 - After activation: shape still (batch_size, seq_len, intermediate_size)
    4 - After `dense_2`: (batch_size, seq_len, hidden_size)
    5 - After dropout: shape still (batch_size, seq_len, hidden_size)
    """

    def __init__(self, config):
        super().__init__()
        # The first linear layer increases the dimension from hidden_size to intermediate_size 
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        
        # Activation function, a variant of GELU
        self.activation = NewGELUActivation()

        # The second linear layer brings dimension back from intermediate_size to hidden_size
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

        # A dropout for regularization
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        """
        Args:
            x: shape => (batch_size, sequence_length, hidden_size)

        Returns:
            shape => (batch_size, sequence_length, hidden_size)
        """
        # 1) project from hidden_size -> intermediate_size
        x = self.dense_1(x)         # => shape (batch_size, sequence_length, intermediate_size)
        
        # 2) apply the chosen activation function 
        x = self.activation(x)
        
        # 3) project from intermediate_size -> hidden_size
        x = self.dense_2(x)         # => shape (batch_size, sequence_length, hidden_size)

        # 4) apply dropout
        x = self.dropout(x)
        
        return x


class Block(nn.Module):
    """
    A single transformer block.

    This module represents one Transformer layer and integrates multi-head attention + MLP + skip-connection logic.
    """

    def __init__(self, config):
        super().__init__()

        # Check if we want to use the "FastMultiHeadAttention" or the standard "MultiHeadAttention" 
        self.use_faster_attention = config.get("use_faster_attention", False)
        
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        
        # LayerNorm is placed before attention and MLP in a "pre-norm" style
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        
        # The feed-forward network module
        self.mlp = MLP(config)
        
        # Another LayerNorm before the MLP
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        """
        Args:
            x: shape => (batch_size, sequence_length, hidden_size)
            output_attentions: bool, whether to return the attention weights

        Returns:
            (x, attention_probs) or (x, None)
            - x is the updated hidden states => (batch_size, sequence_length, hidden_size)
            - attention_probs => (batch_size, num_heads, seq_length, seq_length) or None
        """

        # 1) LayerNorm before self-attention (pre-norm)
        #   shape => (batch_size, sequence_length, hidden_size)
        normed_x = self.layernorm_1(x)

        # 2) Self-attention
        attention_output, attention_probs = self.attention(
            normed_x, output_attentions=output_attentions
        )
        
        # 3) Add & Norm (the "skip connection" a.k.a residual connection)
        #    We add the attention_output to the original x
        x = x + attention_output
    
        # 4) The we do the feed-forward part (MLP)
        #   First do a seconnd layernorm
        normed_x2 = self.layernorm_2(x)

        #   MLP => shape remains (batch_size, sequence_length, hidden_size)
        mlp_output = self.mlp(normed_x2)
        
        # 5) Another skip connection
        x = x + mlp_output
        
        # 6) Return the final updated representation and the attention weights (optionally)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.

    - The Encoder is a stack of Transformer blocks (Block class)
    - It processes the input sequence (image patches + CLS token) through multiple layers.
    - The sequence length remains the same throughout.

    In the paper, there are several number of layers (12, 24, 32) corresponding to the ViT-Base, ViT-Large, ViT-Huge models.
    """
    def __init__(self, config):
        super().__init__()
        
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        
        # The number of transformer layers
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        """
        Args:
            x: shape => (batch_size, sequence_length, hidden_size)
            output_attentions: if True, return attention maps from all layers

        Returns:
            - final_hidden_state: shape (batch_size, sequence_length, hidden_size)
            - all_attentions (optional): list of attention maps from all blocks
        """
        
        # Store attention probabilities if required
        all_attentions = []
        
        # Forward pass through each transformer block
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            
            # If output_attentions is True, store the attention probabilities
            if output_attentions:
                all_attentions.append(attention_probs)
        
        # Return the final hidden representation and attention maps (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    The ViT model for image classification.

    Shape evolution
    1 - Input: batch of images - (batch_size, channels, H, W)
    2 - Embedding: (batch_size, seq_len, hidden_size) with seq_len = num_patches + 1
    3 - Encoder: (batch_size, seq_len, hidden_size)
    4 - Classifier:
        - Take out the [CLS] token => shape (batch_size, hidden_size)
        - `classifier` => shape (batch_size, num_classes)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        
        # 1) Create the embedding module
        self.embedding = Embeddings(config)
        
        # 2) Create the transformer encoder module
        #    This is the stack of Transformer blocks (self-attention + MLP + skip connections)
        self.encoder = Encoder(config)
        
        # 3) Classification head (linear):
        #    - We take the [CLS] token output from the encoder -> shape: (batch_size, hidden_size)
        #    - Then we map it to (batch_size, num_classes)
        # Create a linear layer to project the encoder's output to the number of classes
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        
        # 4) Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        """
        Args:
            x: A batch of images => shape (batch_size, image_size, image_size)
            output_attentions: bool. If True, we also return the attention maps from the encoder.

        Returns:
            A tuple (logits, all_attentions or None)
            - logits => shape (batch_size, num_classes)
            - all_attentions => either None or list of attention maps (one per Transformer block)
        """

        # a) Convert images to patch+position embeddings => (batch_size, seq_len, hidden_size)
        embedding_output = self.embedding(x)
        
        # b) Pass embeddings through the Transformer encoder => (batch_size, seq_len, hidden_size)
        #       plus optional list of attention maps
        encoder_output, all_attentions = self.encoder(
            embedding_output, 
            output_attentions=output_attentions
        )
        
        # c) Extract the [CLS] token representation from the first position => (batch_size, hidden_size)
        cls_representation = encoder_output[:, 0, :]
        
        # d) Classifier => (batch_size, num_classes)
        logits = self.classifier(cls_representation)
        
        # e) Return the logits and (optionally) the attention maps
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        """
        Custom initialization for the layers

        - Set up a normal distribution for linear/conv layers.
        - For embeddings, sets the position embeddings and [CLS] token to a truncated normal distribution.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Normal distribution with mean=0, std=initializer_range
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            # Truncated normal initialization for position_embeddings & cls_token
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)