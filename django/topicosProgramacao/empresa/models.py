# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class Pessoa(models.Model):
    AGUARDANDO_VAGA = '2'
    CONTRATADO = '1'
    DEMITIDO = '0'

    ESTADO_FUNCIONAL_CHOICES = (
        (AGUARDANDO_VAGA, 'Aguardando Vaga'),
        (CONTRATADO, 'Contratado'),
        (DEMITIDO, 'Demitido'),
    )

    id_funcional = models.PositiveIntegerField(
        unique=True,
    )
    estado_funcional = models.CharField(
        max_length=1,
        choices=ESTADO_FUNCIONAL_CHOICES,
        default=AGUARDANDO_VAGA
    )

    nome = models.CharField(
        max_length=100,
        blank=False,
        unique=True,
    )

    profissao = models.CharField(max_length=100)
    endereco = models.CharField(max_length=100)
    funcao = models.CharField(max_length=100)
    cargo = models.CharField(max_length=100)

    # TODO options da faixa salarial
    faixa_salarial = models.CharField(max_length=2)

    def __str__(self):
        return self.id, ': ', self.nome

class Empresa(models.Model):
    nome = models.CharField(
        max_length=100,
        blank=False,
        unique=True,
    )
    max_funcionarios = models.PositiveIntegerField(
        unique=True,
        default=5,
    )

    def __str__(self):
        return self.nome

class Funcionario(models.Model):
    pessoa = models.OneToOneField(
        Pessoa,
        on_delete=models.CASCADE,
        blank=False,
    )

    empresa = models.ForeignKey(Empresa)

    def __str__(self):
        return self.id, ': ', self.pessoa.nome