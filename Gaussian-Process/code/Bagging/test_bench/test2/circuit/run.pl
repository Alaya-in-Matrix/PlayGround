#!/usr/bin/perl
use strict;
use warnings;
use 5.010;
`./clear.sh`;
`rm -f result.po`;
`hspice64 ./Single_ended.sp -o dac2014`;

my %data;
$data{gain} = 0;
$data{pm}   = 0;
$data{ugf}  = 0;

open my $ifh, "<", "dac2014.ma0", or die "Can't open dac2014.ma0$!\n";
chomp(my @contents =  <$ifh>);
close $ifh;

open my $ofh, ">", "result.po";
if($contents[4] =~ /^\s+(.*?)\s+(.*?)\s+(.*?)\s+(.*)\s+/)
{
    my $neg_gain = -1 * $1;
    my $neg_ugf  = -1 * $2 * 1e-6;
    my $neg_pm   = -1 * $4;
    say $ofh "$neg_gain $neg_ugf $neg_pm";
}
close $ofh;

`cat param result.po >> backup`;
