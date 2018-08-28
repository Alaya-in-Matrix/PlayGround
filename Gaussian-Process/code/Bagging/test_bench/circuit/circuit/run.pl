#!/usr/bin/perl
use strict;
use warnings;
use 5.010;

run_cmd("rm -f *0 *.info TT.sp FF.sp SS.sp SF.sp FS.cpp");
run_cmd("rm -f *.mt* *.ma* *.ms*");
run_cmd("rm -f TT.*");
run_cmd("cat ./head/TT_head.sp Single_ended.sp > TT.sp");
run_cmd("hspicerf64 ./TT.sp > TT.info 2>&1");

my %TT;
extract(\%TT, "TT");

# name | type| TT    | SS   | FF    | SF   | FS
# ugf  |  >  | 0.92  | 0.78 | 1.08  | 1.04 | 0.82
# pm   |  >  | 52.5  | 55.6 | 50.3  | 50.6 | 55.2
# gm   |  >  | 19.5  | 19.3 | 20.2  | 19.1 | 20.3
# srr  |  >  | 0.18  | 0.14 | 0.26  | 0.21 | 0.16
# srf  |  >  | 0.2   | 0.15 | 0.27  | 0.16 | 0.25
# iq   |  <  | 60.7  | 63.2 | 78.6  | 78.0 | 63.7

my $obj      = $TT{iq};
my $vio_gain = 100  - $TT{gain}; # gain > 100
my $vio_ugf  = 0.92 - $TT{ugf}; # ugf > 0.92
my $vio_pm   = 52.5 - $TT{pm};  # pm  > 52.5
my $vio_gm   = 19.5 - $TT{gm};  # gm  > 19.5
my $vio_srr  = 0.18 - $TT{srr}; # srr > 0.18
my $vio_srf  = 0.2  - $TT{srf}; # srf > 0.2

open my $ofh, ">", "result.po" or die "can't create result.po:$!\n";
# say $ofh "Iq vio_gain vio_ugf vio_pm vio_gm vio_srr vio_srf";
say $ofh "$obj $vio_gain $vio_ugf $vio_pm $vio_gm $vio_srr $vio_srf";
close $ofh;

sub run_cmd
{
    my $cmd = shift;
    my $ret = system($cmd);
    die "Fail to run $cmd\n" if($ret != 0);
}

sub extract
{
    # gain pm ugf srr srf iq
    my ($rh, $corner) = @_;
    my $file    = "$corner.lis";
    $rh->{gain} = 0;
    $rh->{pm}   = 0;
    $rh->{gm}   = 0;
    $rh->{ugf}  = 0;
    $rh->{srr}  = 0;
    $rh->{srf}  = 0;
    $rh->{iq}   = 0;
    open my $fh, "<", $file or die "Can't read $file: $!\n";
    my %onfail;
    $onfail{gain} = sprintf("%f", "-inf");
    $onfail{pm}   = sprintf("%f", "0");
    $onfail{gm}   = sprintf("%f", "0");
    $onfail{ugf}  = sprintf("%f", "0");
    $onfail{srr}  = sprintf("%f", "-inf");
    $onfail{srf}  = sprintf("%f", "-inf");
    $onfail{iq}   = sprintf("%f", "inf");
    while(my $line = <$fh>)
    {
        chomp($line);
        my @tok = keys %{$rh};
        for my $name (@tok)
        {
            if($line =~ /$name=\s+(.*?)\s+/)
            {
                my $val = $1;
                if($val =~ /failed/)
                {
                    $rh->{$name} = $onfail{$name};
                }
                else
                {
                    $rh->{$name} = sprintf("%.18f", $val);
                }
            }
        }
    }
    close $fh;
}
sub max
{
    my($a, $b) = @_;
    return $a > $b ? $a : $b;
}
